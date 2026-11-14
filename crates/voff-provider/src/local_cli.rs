//! Local coding-agent CLI adapters.
//!
//! These adapters use the CLI's existing login. They intentionally do not
//! resolve, require, or inherit API-key environment variables.
use crate::{adapters::cancelled, adapters::send, *};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    env,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::Command,
};
use tokio_util::sync::CancellationToken;
use voff_prompts::{
    provider::{LOCAL_CLI_PREFIX, LOCAL_CLI_SCHEMA_SUFFIX},
    tool_descriptions,
};

const MAX_CLI_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const ALLOWED_ENVIRONMENT: &[&str] = &[
    "PATH",
    "HOME",
    // Account names, not secrets: Claude Code locates its keychain login by
    // them and otherwise reports "Not logged in".
    "USER",
    "LOGNAME",
    "CODEX_HOME",
    "CLAUDE_CONFIG_DIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalCliKind {
    Codex,
    Claude,
    Copilot,
}

impl LocalCliKind {
    fn executable(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Claude => "claude",
            Self::Copilot => "copilot",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Codex => "Local Codex CLI",
            Self::Claude => "Local Claude Code CLI",
            Self::Copilot => "Local GitHub Copilot CLI",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::Codex => "codex_cli",
            Self::Claude => "claude_cli",
            Self::Copilot => "copilot_cli",
        }
    }
}

/// Copilot has no schema flag, so the schema travels in the prompt and the
/// reply is validated by `parse_local_output` like every other local CLI.
const COPILOT_REQUIRED_FLAGS: &[&str] = &[
    "--output-format",
    "--stream",
    "--available-tools",
    "--deny-tool",
    "--disable-builtin-mcps",
    "--no-ask-user",
    "--no-custom-instructions",
    "--disallow-temp-dir",
    "--no-auto-update",
];

/// A tool name that does not exist. An empty `--available-tools` value is
/// ignored by Copilot and leaves every native tool visible.
const COPILOT_NO_TOOLS: &str = "--available-tools=voff_no_native_tools";

/// Deny rules win over every allow rule in Copilot, including allow-all.
const COPILOT_DENY_RULES: &[&str] = &[
    "--deny-tool=shell",
    "--deny-tool=write",
    "--deny-tool=read",
    "--deny-tool=url",
    "--deny-tool=memory",
];

#[derive(Clone, Debug)]
pub struct LocalCliProvider {
    kind: LocalCliKind,
    executable: PathBuf,
    probed: Arc<tokio::sync::OnceCell<()>>,
}

impl LocalCliProvider {
    pub fn new(kind: LocalCliKind, workspace: impl AsRef<Path>) -> Result<Self, ProviderError> {
        let workspace = workspace
            .as_ref()
            .canonicalize()
            .map_err(|error| ProviderError {
                kind: ProviderErrorKind::Authorization,
                message: format!("local CLI workspace is unavailable: {error}"),
                retryable: false,
            })?;
        if !workspace.is_dir() {
            return Err(ProviderError {
                kind: ProviderErrorKind::Authorization,
                message: "local CLI workspace is not a directory".into(),
                retryable: false,
            });
        }
        Ok(Self {
            kind,
            executable: kind.executable().into(),
            probed: Arc::new(tokio::sync::OnceCell::new()),
        })
    }
}

#[async_trait]
impl ModelProvider for LocalCliProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        ProviderDescriptor {
            id: self.kind.id().into(),
            display_name: self.kind.display_name().into(),
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                structured_output: true,
                images: false,
                cancellation: true,
            },
        }
    }

    async fn models(&self) -> Result<Vec<ModelDescriptor>, ProviderError> {
        self.probe(&CancellationToken::new()).await?;
        Ok(vec![ModelDescriptor {
            id: "default".into(),
            display_name: "CLI default".into(),
            context_limit: None,
        }])
    }

    async fn start(
        &self,
        request: CanonicalRequest,
        sink: ProviderEventSink,
        cancellation: CancellationToken,
    ) -> Result<ProviderCompletion, ProviderError> {
        self.probe(&cancellation).await?;
        let (schema, tools) = output_schema(&request)?;
        let prompt = local_prompt(&request, &tools)?;
        let tool_required = matches!(request.tool_choice, ToolChoice::Specific(_));
        let (output, usage) = match self.kind {
            LocalCliKind::Codex => {
                self.run_codex(&request.model, &prompt, &schema, &sink, &cancellation)
                    .await
            }
            LocalCliKind::Claude => {
                self.run_claude(&request.model, &prompt, &schema, &sink, &cancellation)
                    .await
            }
            LocalCliKind::Copilot => {
                self.run_copilot(&request.model, &prompt, &schema, &sink, &cancellation)
                    .await
            }
        }
        .inspect_err(|error| {
            tracing::warn!(
                task_id = %request.task_id,
                provider = self.kind.id(),
                kind = ?error.kind,
                diagnostic = %error.message,
                "local CLI request failed"
            );
        })?;
        let output = parse_local_output(&output, &tools, tool_required).inspect_err(|error| {
            tracing::warn!(
                task_id = %request.task_id,
                provider = self.kind.id(),
                kind = ?error.kind,
                diagnostic = %error.message,
                "local CLI response rejected"
            );
        })?;
        tracing::info!(task_id = %request.task_id, provider = self.kind.id(), tool_calls = output.tool_calls.len(), "local CLI response decoded");
        if output.text.is_empty() && output.tool_calls.is_empty() {
            return Err(invalid_cli_response("local CLI returned no usable output"));
        }
        if let Some(usage) = &usage {
            send(&sink, ProviderEvent::Usage(usage.clone()), &cancellation).await?;
        }
        if !output.text.is_empty() {
            send(&sink, ProviderEvent::TextDelta(output.text), &cancellation).await?;
        }
        for call in output.tool_calls {
            send(
                &sink,
                ProviderEvent::ToolRequest(StructuredToolRequest {
                    id: voff_core::ToolCallId::new(),
                    name: call.name,
                    arguments_json: call.arguments.to_string(),
                }),
                &cancellation,
            )
            .await?;
        }
        send(&sink, ProviderEvent::Completed, &cancellation).await?;
        Ok(ProviderCompletion { usage })
    }
}

impl LocalCliProvider {
    async fn probe(&self, cancellation: &CancellationToken) -> Result<(), ProviderError> {
        self.probed
            .get_or_try_init(|| self.probe_uncached(cancellation))
            .await
            .map(|_| ())
    }

    async fn probe_uncached(&self, cancellation: &CancellationToken) -> Result<(), ProviderError> {
        let execution_directory = tempfile::tempdir().map_err(local_io_error)?;
        let mut command = Command::new(&self.executable);
        command.current_dir(execution_directory.path());
        match self.kind {
            LocalCliKind::Codex => {
                command.args(["app-server", "--help"]);
            }
            LocalCliKind::Claude | LocalCliKind::Copilot => {
                command.arg("--help");
            }
        }
        configure_environment(&mut command);
        if self.kind == LocalCliKind::Copilot {
            configure_copilot_environment(&mut command, execution_directory.path());
        }
        let output =
            tokio::time::timeout(PROBE_TIMEOUT, run_process(command, b"", cancellation, None))
                .await
                .map_err(|_| ProviderError {
                    kind: ProviderErrorKind::Timeout,
                    message: format!("{} capability probe timed out", self.executable.display()),
                    retryable: false,
                })??;
        let help = String::from_utf8_lossy(&output);
        let required: &[&str] = match self.kind {
            LocalCliKind::Codex => &["--listen"],
            LocalCliKind::Claude => &[
                "--json-schema",
                "--restricted",
                "--safe-mode",
                "--permission-prompts",
                "--no-session-persistence",
                "--include-partial-messages",
            ],
            LocalCliKind::Copilot => COPILOT_REQUIRED_FLAGS,
        };
        let missing = required
            .iter()
            .filter(|flag| !help.contains(**flag))
            .copied()
            .collect::<Vec<_>>();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(ProviderError {
                kind: ProviderErrorKind::Authorization,
                message: format!(
                    "{} is incompatible; update it to a release supporting {}",
                    self.executable.display(),
                    missing.join(", ")
                ),
                retryable: false,
            })
        }
    }

    async fn run_codex(
        &self,
        model: &str,
        prompt: &[u8],
        schema: &Value,
        sink: &ProviderEventSink,
        cancellation: &CancellationToken,
    ) -> Result<(Vec<u8>, Option<Usage>), ProviderError> {
        crate::codex_server::run(&self.executable, model, prompt, schema, sink, cancellation).await
    }

    async fn run_claude(
        &self,
        model: &str,
        prompt: &[u8],
        schema: &Value,
        sink: &ProviderEventSink,
        cancellation: &CancellationToken,
    ) -> Result<(Vec<u8>, Option<Usage>), ProviderError> {
        let schema = serde_json::to_string(schema).map_err(|error| {
            invalid_cli_response(&format!("could not encode local output schema: {error}"))
        })?;
        let execution_directory = tempfile::tempdir().map_err(local_io_error)?;
        let mut command = Command::new(&self.executable);
        command.current_dir(execution_directory.path()).args([
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--json-schema",
            &schema,
            "--restricted",
            "--safe-mode",
            "--tools",
            "",
            "--disallowedTools",
            "mcp__*",
            "--permission-prompts",
            "none",
            "--no-session-persistence",
            "--no-chrome",
        ]);
        if model != "default" {
            command.arg("--model").arg(model);
        }
        configure_environment(&mut command);
        let envelope = run_process(
            command,
            prompt,
            cancellation,
            Some((StreamFormat::Claude, sink)),
        )
        .await?;
        let envelope: Value = serde_json::from_slice(&envelope).map_err(|error| {
            invalid_cli_response(&format!("Claude CLI returned invalid JSON: {error}"))
        })?;
        let usage = claude_usage(&envelope["usage"]);
        let output = serde_json::to_vec(
            envelope
                .get("structured_output")
                .ok_or_else(|| invalid_cli_response("Claude CLI omitted structured_output"))?,
        )
        .map_err(|error| invalid_cli_response(&format!("Claude structured output: {error}")))?;
        Ok((output, usage))
    }

    async fn run_copilot(
        &self,
        model: &str,
        prompt: &[u8],
        schema: &Value,
        sink: &ProviderEventSink,
        cancellation: &CancellationToken,
    ) -> Result<(Vec<u8>, Option<Usage>), ProviderError> {
        let mut prompt = prompt.to_vec();
        prompt.extend_from_slice(LOCAL_CLI_SCHEMA_SUFFIX.as_bytes());
        serde_json::to_writer(&mut prompt, schema).map_err(|error| {
            invalid_cli_response(&format!("could not encode local output schema: {error}"))
        })?;
        // The temporary directory is both the working directory and
        // COPILOT_HOME, so user MCP servers, settings, permission grants and
        // saved sessions are neither loaded nor left behind.
        let execution_directory = tempfile::tempdir().map_err(local_io_error)?;
        let mut command = Command::new(&self.executable);
        command
            .current_dir(execution_directory.path())
            .args(copilot_arguments(model));
        configure_environment(&mut command);
        configure_copilot_environment(&mut command, execution_directory.path());
        let message = run_process(
            command,
            &prompt,
            cancellation,
            Some((StreamFormat::Copilot, sink)),
        )
        .await?;
        let message: Value = serde_json::from_slice(&message).map_err(|error| {
            invalid_cli_response(&format!("Copilot CLI returned invalid JSON: {error}"))
        })?;
        let content = message["data"]["content"]
            .as_str()
            .ok_or_else(|| invalid_cli_response("Copilot CLI omitted message content"))?;
        // Copilot bills premium requests rather than tokens; report only the
        // effective model so "auto" selections remain visible.
        let usage = message["data"]["model"].as_str().map(|model| Usage {
            model: Some(model.into()),
            ..Usage::default()
        });
        Ok((extract_json_object(content).as_bytes().to_vec(), usage))
    }
}

fn copilot_arguments(model: &str) -> Vec<String> {
    let mut arguments = [
        "--output-format",
        "json",
        "--stream",
        "on",
        COPILOT_NO_TOOLS,
        "--disable-builtin-mcps",
        "--no-ask-user",
        "--no-custom-instructions",
        "--disallow-temp-dir",
        "--no-auto-update",
        "--disable-client-telemetry",
    ]
    .iter()
    .chain(COPILOT_DENY_RULES)
    .map(|argument| (*argument).to_owned())
    .collect::<Vec<_>>();
    if model != "default" {
        arguments.extend(["--model".into(), model.into()]);
    }
    arguments
}

fn configure_copilot_environment(command: &mut Command, home: &Path) {
    command
        .env("COPILOT_HOME", home)
        .env("COPILOT_AUTO_UPDATE", "false");
}

/// Accepts a bare object or one wrapped in Markdown fences or brief prose.
/// The extracted text is still validated strictly by `parse_local_output`.
fn extract_json_object(content: &str) -> &str {
    let trimmed = content.trim();
    match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(start), Some(end)) if start < end => &trimmed[start..=end],
        _ => trimmed,
    }
}

#[derive(Clone, Copy)]
enum StreamFormat {
    Claude,
    Copilot,
}

fn claude_usage(usage: &Value) -> Option<Usage> {
    let input_tokens = usage["input_tokens"]
        .as_u64()
        .unwrap_or_default()
        .saturating_add(
            usage["cache_creation_input_tokens"]
                .as_u64()
                .unwrap_or_default(),
        )
        .saturating_add(
            usage["cache_read_input_tokens"]
                .as_u64()
                .unwrap_or_default(),
        );
    let output_tokens = usage["output_tokens"].as_u64().unwrap_or_default();
    (input_tokens > 0 || output_tokens > 0).then_some(Usage {
        input_tokens,
        output_tokens,
        ..Usage::default()
    })
}

pub(super) fn configure_environment(command: &mut Command) {
    command
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for key in ALLOWED_ENVIRONMENT {
        if let Some(value) = env::var_os(key) {
            command.env(key, value);
        }
    }
}

async fn run_process(
    mut command: Command,
    prompt: &[u8],
    cancellation: &CancellationToken,
    preview: Option<(StreamFormat, &ProviderEventSink)>,
) -> Result<Vec<u8>, ProviderError> {
    let program = command
        .as_std()
        .get_program()
        .to_string_lossy()
        .into_owned();
    let mut child = command.spawn().map_err(|error| ProviderError {
        kind: ProviderErrorKind::Authorization,
        message: format!("could not start {program}: {error}"),
        retryable: false,
    })?;
    let mut stdin = child.stdin.take().ok_or_else(|| ProviderError {
        kind: ProviderErrorKind::Internal,
        message: format!("{program} stdin was unavailable"),
        retryable: false,
    })?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let io = async {
        let write = async {
            stdin.write_all(prompt).await.map_err(local_io_error)?;
            stdin.shutdown().await.map_err(local_io_error)?;
            drop(stdin);
            Ok::<(), ProviderError>(())
        };
        let read_stdout = async {
            match (stdout, preview) {
                (Some(stdout), Some((StreamFormat::Claude, sink))) => {
                    read_claude_stream(stdout, sink, cancellation).await
                }
                (Some(stdout), Some((StreamFormat::Copilot, sink))) => {
                    read_copilot_stream(stdout, sink, cancellation).await
                }
                (stdout, _) => read_pipe(stdout).await,
            }
        };
        let read_stderr = read_pipe(stderr);
        let wait = async { child.wait().await.map_err(local_io_error) };
        let ((), output, diagnostic, status) =
            tokio::try_join!(write, read_stdout, read_stderr, wait)?;
        if !status.success() {
            return Err(classify_process_failure(&program, status, &diagnostic));
        }
        Ok(output)
    };
    tokio::select! {
        biased;
        () = cancellation.cancelled() => Err(cancelled()),
        () = tokio::time::sleep(Duration::from_secs(300)) => Err(ProviderError {
            kind: ProviderErrorKind::Timeout,
            message: "local CLI exceeded the five-minute request deadline".into(),
            retryable: true,
        }),
        result = io => result,
    }
}

async fn read_claude_stream<R: tokio::io::AsyncRead + Unpin>(
    stdout: R,
    sink: &ProviderEventSink,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, ProviderError> {
    let mut reader = BufReader::new(stdout);
    let mut total = 0usize;
    let mut result = None;
    loop {
        let mut line = Vec::new();
        let size = (&mut reader)
            .take((MAX_CLI_OUTPUT_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .await
            .map_err(local_io_error)?;
        if size == 0 {
            break;
        }
        total = total.saturating_add(size);
        if size > MAX_CLI_OUTPUT_BYTES || total > 16 * MAX_CLI_OUTPUT_BYTES {
            return Err(invalid_cli_response(
                "Claude stream exceeded bounded output limits",
            ));
        }
        let value: Value = serde_json::from_slice(&line)
            .map_err(|_| invalid_cli_response("Claude returned malformed stream JSON"))?;
        match value["type"].as_str() {
            Some("stream_event") if result.is_none() => {
                let delta = &value["event"]["delta"];
                let text = match delta["type"].as_str() {
                    Some("text_delta") => delta["text"].as_str(),
                    Some("input_json_delta") => delta["partial_json"].as_str(),
                    _ => None,
                };
                if let Some(text) = text {
                    send(sink, ProviderEvent::OutputDelta(text.into()), cancellation).await?;
                }
            }
            Some("result") => {
                if result.is_some() {
                    return Err(invalid_cli_response(
                        "Claude returned multiple terminal results",
                    ));
                }
                if value["is_error"] == true || value["subtype"] != "success" {
                    // Match only the CLI's own login status; never echo the
                    // result text, which may contain model or prompt content.
                    if value["result"]
                        .as_str()
                        .is_some_and(|result| result.to_ascii_lowercase().contains("not logged in"))
                    {
                        return Err(ProviderError {
                            kind: ProviderErrorKind::Authentication,
                            message: "Claude CLI is not logged in; run `claude` and use /login, then retry".into(),
                            retryable: false,
                        });
                    }
                    return Err(invalid_cli_response(
                        "Claude failed its structured turn; check login, model availability and limits",
                    ));
                }
                if value.get("structured_output").is_none() {
                    return Err(invalid_cli_response("Claude omitted structured_output"));
                }
                result = Some(line);
            }
            _ => {} // Metadata and reasoning are never interpreted as Voff actions.
        }
    }
    result.ok_or_else(|| ProviderError {
        kind: ProviderErrorKind::Internal,
        message: "Claude stream ended before a terminal result".into(),
        retryable: true,
    })
}

/// Returns the final `assistant.message` line once a successful `result`
/// arrives. Any native tool activity fails the turn: Voff never lets a local
/// CLI act on the workspace directly.
async fn read_copilot_stream<R: tokio::io::AsyncRead + Unpin>(
    stdout: R,
    sink: &ProviderEventSink,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, ProviderError> {
    let mut reader = BufReader::new(stdout);
    let mut total = 0usize;
    let mut message = None;
    let mut finished = false;
    loop {
        let mut line = Vec::new();
        let size = (&mut reader)
            .take((MAX_CLI_OUTPUT_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .await
            .map_err(local_io_error)?;
        if size == 0 {
            break;
        }
        total = total.saturating_add(size);
        if size > MAX_CLI_OUTPUT_BYTES || total > 16 * MAX_CLI_OUTPUT_BYTES {
            return Err(invalid_cli_response(
                "Copilot stream exceeded bounded output limits",
            ));
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let value: Value = serde_json::from_slice(&line)
            .map_err(|_| invalid_cli_response("Copilot returned malformed stream JSON"))?;
        let kind = value["type"].as_str().unwrap_or_default();
        if kind.starts_with("tool.") || kind == "assistant.tool_call_delta" {
            return Err(native_tool_attempt());
        }
        match kind {
            _ if finished && kind == "result" => {
                return Err(invalid_cli_response(
                    "Copilot returned multiple terminal results",
                ));
            }
            _ if finished => {} // Trailing metadata cannot change the turn.
            "assistant.message_delta" => {
                if let Some(text) = value["data"]["deltaContent"].as_str() {
                    send(sink, ProviderEvent::OutputDelta(text.into()), cancellation).await?;
                }
            }
            "assistant.message" => {
                if value["data"]["toolRequests"]
                    .as_array()
                    .is_some_and(|requests| !requests.is_empty())
                {
                    return Err(native_tool_attempt());
                }
                message = Some(line);
            }
            "result" => {
                if value["exitCode"].as_i64() != Some(0) {
                    return Err(invalid_cli_response(
                        "Copilot failed its turn; check login, model availability and limits",
                    ));
                }
                finished = true;
            }
            _ => {} // Metadata and reasoning are never interpreted as Voff actions.
        }
    }
    if !finished {
        return Err(ProviderError {
            kind: ProviderErrorKind::Internal,
            message: "Copilot stream ended before a terminal result".into(),
            retryable: true,
        });
    }
    message.ok_or_else(|| invalid_cli_response("Copilot finished without an assistant message"))
}

fn native_tool_attempt() -> ProviderError {
    invalid_cli_response("Copilot attempted a native tool; Voff only accepts proposed Voff tools")
}

pub(super) async fn read_pipe<R: tokio::io::AsyncRead + Unpin>(
    pipe: Option<R>,
) -> Result<Vec<u8>, ProviderError> {
    let mut bytes = Vec::new();
    if let Some(pipe) = pipe {
        pipe.take((MAX_CLI_OUTPUT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .await
            .map_err(local_io_error)?;
    }
    if bytes.len() > MAX_CLI_OUTPUT_BYTES {
        return Err(invalid_cli_response(
            "local CLI output exceeded 2097152 bytes",
        ));
    }
    Ok(bytes)
}

fn classify_process_failure(
    program: &str,
    status: std::process::ExitStatus,
    diagnostic: &[u8],
) -> ProviderError {
    let diagnostic = String::from_utf8_lossy(diagnostic).to_ascii_lowercase();
    let (kind, retryable, action) =
        if diagnostic.contains("invalid_json_schema") || diagnostic.contains("invalid schema") {
            (
                ProviderErrorKind::InvalidResponse,
                false,
                "rejected voff's structured output schema; update voff or the CLI",
            )
        } else if diagnostic.contains("not logged in")
            || diagnostic.contains("unauthorized")
            || diagnostic.contains("authentication")
            || diagnostic.contains("status: 401")
        {
            (
                ProviderErrorKind::Authentication,
                false,
                "is not logged in; authenticate with the CLI and retry",
            )
        } else if diagnostic.contains("rate_limit")
            || diagnostic.contains("rate limit")
            || diagnostic.contains("status: 429")
            || diagnostic.contains("too many requests")
        {
            (
                ProviderErrorKind::RateLimit,
                true,
                "was rate limited; retry after the provider cooldown",
            )
        } else if diagnostic.contains("unknown argument")
            || diagnostic.contains("unexpected argument")
            || diagnostic.contains("unrecognized option")
        {
            (
                ProviderErrorKind::Authorization,
                false,
                "does not support the required safe CLI flags; update the CLI",
            )
        } else if diagnostic.contains("internal server")
            || diagnostic.contains("server error")
            || diagnostic.contains("service unavailable")
            || diagnostic.contains("temporarily unavailable")
            || diagnostic.contains("temporary failure")
            || diagnostic.contains("try again later")
            || diagnostic.contains("overloaded")
            || diagnostic.contains("gateway timeout")
            || diagnostic.contains("bad gateway")
            || ["status: 500", "status: 502", "status: 503", "status: 504"]
                .iter()
                .any(|marker| diagnostic.contains(marker))
        {
            (
                ProviderErrorKind::Internal,
                true,
                "hit a temporary upstream error; retrying the request",
            )
        } else if diagnostic.contains("model_not_found")
            || diagnostic.contains("model not found")
            || diagnostic.contains("does not exist")
            || (diagnostic.contains("model") && diagnostic.contains("is not available"))
        {
            (
                ProviderErrorKind::InvalidResponse,
                false,
                "cannot use the configured model; choose an available CLI model",
            )
        } else {
            (
                ProviderErrorKind::InvalidResponse,
                false,
                "failed; verify the CLI installation, login, and selected model",
            )
        };
    ProviderError {
        kind,
        message: format!("{program} {action} ({status})"),
        retryable,
    }
}

pub(super) fn local_io_error(error: std::io::Error) -> ProviderError {
    ProviderError {
        kind: ProviderErrorKind::Internal,
        message: format!("local CLI I/O failed: {error}"),
        retryable: false,
    }
}

fn invalid_cli_response(message: &str) -> ProviderError {
    ProviderError {
        kind: ProviderErrorKind::InvalidResponse,
        message: message.into(),
        retryable: false,
    }
}

fn selected_tool_names(request: &CanonicalRequest) -> Result<Vec<String>, ProviderError> {
    let names = match &request.tool_choice {
        ToolChoice::Auto => request.tools.iter().map(|tool| tool.name.clone()).collect(),
        ToolChoice::Specific(names) => names
            .iter()
            .map(|name| {
                request
                    .tools
                    .iter()
                    .find(|tool| tool.name == *name)
                    .map(|tool| tool.name.clone())
                    .ok_or_else(|| invalid_cli_response(&format!("unknown required tool {name}")))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok(names)
}

fn output_schema(request: &CanonicalRequest) -> Result<(Value, Vec<String>), ProviderError> {
    let names = selected_tool_names(request)?;
    let required = matches!(request.tool_choice, ToolChoice::Specific(_));
    let call = if names.is_empty() {
        json!({
            "type":"object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        })
    } else {
        json!({
            "type": "object",
            "properties": {
                "name": {"type":"string", "enum": names},
                "arguments": {
                    "type":"string",
                    "description":tool_descriptions::JSON_TOOL_ARGUMENTS
                }
            },
            "required": ["name", "arguments"],
            "additionalProperties": false
        })
    };
    let mut calls = json!({
        "type": "array",
        "items": call,
        "maxItems": if names.is_empty() { 0 } else { 1 }
    });
    if required {
        calls["minItems"] = 1.into();
    }
    Ok((
        json!({
            "type": "object",
            "properties": {
                "text": {"type":"string"},
                "tool_calls": calls
            },
            "required": ["text", "tool_calls"],
            "additionalProperties": false
        }),
        names,
    ))
}

fn local_prompt(
    request: &CanonicalRequest,
    tool_names: &[String],
) -> Result<Vec<u8>, ProviderError> {
    let tools = request
        .tools
        .iter()
        .filter(|tool| tool_names.contains(&tool.name))
        .map(|tool| {
            let schema = serde_json::from_str::<Value>(&tool.input_schema).map_err(|error| {
                invalid_cli_response(&format!("invalid tool schema for {}: {error}", tool.name))
            })?;
            Ok(json!({"name":tool.name,"description":tool.description,"input_schema":schema}))
        })
        .collect::<Result<Vec<_>, ProviderError>>()?;
    let payload = json!({
        "messages": request.messages,
        "available_voff_tools": tools,
        "tool_choice": match &request.tool_choice {
            ToolChoice::Auto => json!({"mode":"auto"}),
            ToolChoice::Specific(names) => json!({"mode":"required","names":names}),
        },
        "max_output_tokens": request.max_output_tokens,
    });
    let mut prompt = LOCAL_CLI_PREFIX.as_bytes().to_vec();
    serde_json::to_writer(&mut prompt, &payload).map_err(|error| {
        invalid_cli_response(&format!("could not encode local CLI request: {error}"))
    })?;
    Ok(prompt)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalOutput {
    text: String,
    tool_calls: Vec<LocalToolCall>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalToolCall {
    name: String,
    #[serde(deserialize_with = "deserialize_arguments")]
    arguments: Value,
}

fn deserialize_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    let encoded = String::deserialize(deserializer)?;
    let arguments: Value = serde_json::from_str(&encoded).map_err(D::Error::custom)?;
    if !arguments.is_object() {
        return Err(D::Error::custom("tool arguments JSON was not an object"));
    }
    Ok(arguments)
}

fn parse_local_output(
    bytes: &[u8],
    allowed_tools: &[String],
    tool_required: bool,
) -> Result<LocalOutput, ProviderError> {
    let output: LocalOutput = serde_json::from_slice(bytes).map_err(|error| {
        invalid_cli_response(&format!("local CLI output did not match schema: {error}"))
    })?;
    if output.tool_calls.len() > 1 {
        return Err(invalid_cli_response(
            "local CLI returned multiple parallel tool calls",
        ));
    }
    if tool_required && output.tool_calls.is_empty() {
        return Err(invalid_cli_response(
            "local CLI omitted the required tool call",
        ));
    }
    for call in &output.tool_calls {
        if !allowed_tools.contains(&call.name) {
            return Err(invalid_cli_response(
                "local CLI returned an unavailable tool",
            ));
        }
        if !call.arguments.is_object() {
            return Err(invalid_cli_response(
                "local CLI tool arguments were not an object",
            ));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use voff_core::{MessageId, TaskId};

    fn request(choice: ToolChoice) -> CanonicalRequest {
        CanonicalRequest {
            task_id: TaskId::new(),
            model: "default".into(),
            messages: vec![CanonicalMessage {
                id: MessageId::new(),
                role: MessageRole::User,
                content: "inspect this".into(),
            }],
            tools: vec![ToolDefinition {
                name: "file.read".into(),
                description: "Read a file".into(),
                input_schema: r#"{"type":"object","properties":{"path":{"type":"string"}}}"#.into(),
            }],
            tool_choice: choice,
            max_output_tokens: Some(100),
        }
    }

    #[test]
    fn required_tool_is_encoded_in_schema() {
        let (schema, names) =
            output_schema(&request(ToolChoice::Specific(vec!["file.read".into()]))).unwrap();
        assert_eq!(names, ["file.read"]);
        assert_eq!(schema["properties"]["tool_calls"]["minItems"], 1);
        assert_eq!(
            schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"][0],
            "file.read"
        );
        assert_eq!(
            schema["properties"]["tool_calls"]["items"]["properties"]["arguments"]["type"],
            "string"
        );
        assert_eq!(
            schema["properties"]["tool_calls"]["items"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn tool_free_schema_is_still_strict_for_codex() {
        let mut request = request(ToolChoice::Auto);
        request.tools.clear();
        let (schema, _) = output_schema(&request).unwrap();
        let item = &schema["properties"]["tool_calls"]["items"];
        assert_eq!(item["properties"], json!({}));
        assert_eq!(item["required"], json!([]));
        assert_eq!(item["additionalProperties"], false);
    }

    #[test]
    fn local_prompt_distinguishes_native_cli_access_from_voff_tools() {
        let prompt = String::from_utf8(
            local_prompt(&request(ToolChoice::Auto), &["file.read".into()]).unwrap(),
        )
        .unwrap();

        assert!(prompt.contains(
            "Do not use the CLI's native file, shell, network, or subagent capabilities"
        ));
        assert!(prompt.contains("Perform workspace work only by returning tool_calls for tools listed in available_voff_tools"));
        assert!(prompt.contains("Native CLI restrictions do not disable listed Voff tools"));
        assert!(prompt.contains("\"name\":\"file.read\""));
        assert!(!prompt.contains(
            "Do not inspect or modify files, run commands, use network tools, or delegate"
        ));
    }

    #[test]
    fn rejects_tools_outside_the_request() {
        let error = parse_local_output(
            br#"{"text":"","tool_calls":[{"name":"shell.run","arguments":"{}"}]}"#,
            &["file.read".into()],
            false,
        )
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::InvalidResponse);
    }

    #[test]
    fn required_tool_cannot_be_omitted_by_a_local_binary() {
        let error = parse_local_output(
            br#"{"text":"no","tool_calls":[]}"#,
            &["file.read".into()],
            true,
        )
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::InvalidResponse);
    }

    #[test]
    fn encoded_tool_arguments_must_contain_one_json_object() {
        let error = parse_local_output(
            br#"{"text":"","tool_calls":[{"name":"file.read","arguments":"[]"}]}"#,
            &["file.read".into()],
            false,
        )
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::InvalidResponse);
    }

    #[cfg(unix)]
    #[test]
    fn process_failures_are_classified_without_exposing_diagnostics() {
        use std::os::unix::process::ExitStatusExt as _;

        let status = std::process::ExitStatus::from_raw(256);
        let schema = classify_process_failure(
            "codex",
            status,
            b"invalid_json_schema secret prompt contents",
        );
        assert_eq!(schema.kind, ProviderErrorKind::InvalidResponse);
        assert!(schema.message.contains("structured output schema"));
        assert!(!schema.message.contains("secret prompt contents"));

        let rate = classify_process_failure("codex", status, b"status: 429 rate_limit");
        assert_eq!(rate.kind, ProviderErrorKind::RateLimit);
        assert!(rate.retryable);
    }

    async fn copilot_stream(
        lines: &[&str],
    ) -> (Vec<ProviderEvent>, Result<Vec<u8>, ProviderError>) {
        let input = lines.join("\n");
        let (sink, mut receiver) = mpsc::channel(32);
        let result = read_copilot_stream(input.as_bytes(), &sink, &CancellationToken::new()).await;
        drop(sink);
        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }
        (events, result)
    }

    #[tokio::test]
    async fn copilot_stream_previews_deltas_and_keeps_the_final_message() {
        let (events, result) = copilot_stream(&[
            r#"{"type":"assistant.message_delta","data":{"deltaContent":"draft"}}"#,
            r#"{"type":"assistant.message","data":{"content":"first","toolRequests":[]}}"#,
            r#"{"type":"assistant.message","data":{"content":"final","toolRequests":[]}}"#,
            r#"{"type":"result","exitCode":0}"#,
            r#"{"type":"session.shutdown","data":{}}"#,
        ])
        .await;
        let line: Value = serde_json::from_slice(&result.unwrap()).unwrap();
        assert_eq!(line["data"]["content"], "final");
        assert_eq!(events, [ProviderEvent::OutputDelta("draft".into())]);
    }

    #[tokio::test]
    async fn copilot_native_tool_activity_fails_closed() {
        for lines in [
            &[r#"{"type":"tool.execution_start","data":{"toolName":"bash"}}"#][..],
            &[r#"{"type":"assistant.tool_call_delta","data":{}}"#][..],
            &[
                r#"{"type":"assistant.message","data":{"content":"","toolRequests":[{"name":"bash"}]}}"#,
                r#"{"type":"result","exitCode":0}"#,
            ][..],
        ] {
            let error = copilot_stream(lines).await.1.unwrap_err();
            assert_eq!(error.kind, ProviderErrorKind::InvalidResponse);
            assert!(error.message.contains("native tool"));
        }
    }

    #[tokio::test]
    async fn copilot_incomplete_or_failed_turns_are_rejected() {
        let message = r#"{"type":"assistant.message","data":{"content":"{}","toolRequests":[]}}"#;
        let failed = copilot_stream(&[message, r#"{"type":"result","exitCode":1}"#])
            .await
            .1
            .unwrap_err();
        assert_eq!(failed.kind, ProviderErrorKind::InvalidResponse);

        let truncated = copilot_stream(&[message]).await.1.unwrap_err();
        assert_eq!(truncated.kind, ProviderErrorKind::Internal);
        assert!(truncated.retryable);

        let silent = copilot_stream(&[r#"{"type":"result","exitCode":0}"#])
            .await
            .1
            .unwrap_err();
        assert!(silent.message.contains("without an assistant message"));

        let twice = copilot_stream(&[
            message,
            r#"{"type":"result","exitCode":0}"#,
            r#"{"type":"result","exitCode":0}"#,
        ])
        .await
        .1
        .unwrap_err();
        assert!(twice.message.contains("multiple terminal results"));

        let malformed = copilot_stream(&["not json"]).await.1.unwrap_err();
        assert!(malformed.message.contains("malformed"));
    }

    #[test]
    fn copilot_json_is_extracted_from_fences_but_still_strictly_parsed() {
        assert_eq!(
            extract_json_object("```json\n{\"text\":\"a\",\"tool_calls\":[]}\n```"),
            r#"{"text":"a","tool_calls":[]}"#
        );
        assert_eq!(extract_json_object("Here: {\"a\":1} done"), r#"{"a":1}"#);
        assert_eq!(extract_json_object("no object"), "no object");
        assert!(
            parse_local_output(
                extract_json_object("Here: {\"a\":1}").as_bytes(),
                &[],
                false
            )
            .is_err()
        );
    }

    #[test]
    fn copilot_arguments_hide_and_deny_every_native_tool() {
        let arguments = copilot_arguments("default");
        assert!(arguments.contains(&COPILOT_NO_TOOLS.to_owned()));
        assert!(
            !arguments
                .iter()
                .any(|argument| argument == "--available-tools=")
        );
        for rule in COPILOT_DENY_RULES {
            assert!(arguments.contains(&(*rule).to_owned()));
        }
        for unsafe_flag in [
            "--allow-all",
            "--yolo",
            "--allow-all-tools",
            "--allow-all-paths",
            "--allow-all-urls",
            "-p",
        ] {
            assert!(!arguments.iter().any(|argument| argument == unsafe_flag));
        }
        assert!(!arguments.contains(&"--model".to_owned()));
        let pinned = copilot_arguments("gpt-test");
        assert_eq!(pinned[pinned.len() - 2..], ["--model", "gpt-test"]);
    }

    #[test]
    fn copilot_unavailable_model_is_classified() {
        use std::os::unix::process::ExitStatusExt;
        let error = classify_process_failure(
            "copilot",
            std::process::ExitStatus::from_raw(256),
            br#"Error: Model "no-such-model" from --model flag is not available."#,
        );
        assert_eq!(error.kind, ProviderErrorKind::InvalidResponse);
        assert!(error.message.contains("configured model"));
    }

    #[test]
    fn api_key_variables_are_never_inherited() {
        assert!(
            !ALLOWED_ENVIRONMENT
                .iter()
                .any(|key| key.contains("API_KEY"))
        );
        assert!(!ALLOWED_ENVIRONMENT.contains(&"ANTHROPIC_AUTH_TOKEN"));
    }

    #[test]
    fn account_identity_reaches_cli_keychain_lookups() {
        // Claude Code finds its macOS keychain login by account name; without
        // USER it reports "Not logged in" even when the user is logged in.
        assert!(ALLOWED_ENVIRONMENT.contains(&"USER"));
        assert!(ALLOWED_ENVIRONMENT.contains(&"LOGNAME"));
    }

    async fn claude_stream(line: &str) -> Result<Vec<u8>, ProviderError> {
        let (sink, _receiver) = mpsc::channel(8);
        read_claude_stream(line.as_bytes(), &sink, &CancellationToken::new()).await
    }

    #[tokio::test]
    async fn claude_logged_out_turns_are_authentication_failures() {
        let error = claude_stream(
            r#"{"type":"result","subtype":"success","is_error":true,"result":"Not logged in · Please run /login"}"#,
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, ProviderErrorKind::Authentication);
        assert!(!error.retryable);
        assert!(error.message.contains("not logged in"));

        let other = claude_stream(
            r#"{"type":"result","subtype":"error_during_execution","is_error":true,"result":"secret prompt echo"}"#,
        )
        .await
        .unwrap_err();
        assert_eq!(other.kind, ProviderErrorKind::InvalidResponse);
        assert!(!other.message.contains("secret prompt echo"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn exact_cli_processes_are_probed_and_schema_validated() {
        let directory = tempfile::tempdir().unwrap();
        for (kind, script, expected_usage) in [
            (
                LocalCliKind::Codex,
                r#"#!/bin/sh
if [ "$2" = "--help" ]; then
  printf '%s\n' '--listen'
  exit 0
fi
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*) printf '%s\n' '{"id":1,"result":{}}' ;;
    *'"method":"config/read"'*) printf '%s\n' '{"id":2,"result":{"config":{"notify":[],"web_search":"disabled","features":{"hooks":false,"shell_tool":false,"plugins":false,"enable_mcp_apps":false,"unified_exec":false,"shell_snapshot":false,"multi_agent":false,"multi_agent_v2":false,"code_mode":false,"js_repl":false,"memories":false,"view_image":false}}}}' ;;
    *'"method":"thread/start"'*) printf '%s\n' '{"id":3,"result":{"thread":{"id":"thread-test","ephemeral":true},"approvalPolicy":"never","sandbox":{"type":"readOnly"}}}' ;;
    *'"method":"turn/start"'*)
      printf '%s\n' '{"id":4,"result":{"turn":{"id":"turn-test"}}}' '{"method":"item/agentMessage/delta","params":{"threadId":"thread-test","turnId":"turn-test","delta":"{\"text\":\"codex ok\",\"tool_calls\":[]}"}}' '{"method":"item/completed","params":{"threadId":"thread-test","turnId":"turn-test","item":{"type":"agentMessage","text":"{\"text\":\"codex ok\",\"tool_calls\":[]}"}}}' '{"method":"thread/tokenUsage/updated","params":{"threadId":"thread-test","turnId":"turn-test","tokenUsage":{"last":{"inputTokens":8,"cachedInputTokens":1,"outputTokens":3,"reasoningOutputTokens":0,"totalTokens":11},"total":{"inputTokens":8,"cachedInputTokens":1,"outputTokens":3,"reasoningOutputTokens":0,"totalTokens":11}}}}' '{"method":"thread/tokenUsage/updated","params":{"threadId":"thread-test","turnId":"turn-test","tokenUsage":{"last":{"inputTokens":12,"cachedInputTokens":3,"outputTokens":7,"reasoningOutputTokens":0,"totalTokens":19},"total":{"inputTokens":12,"cachedInputTokens":3,"outputTokens":7,"reasoningOutputTokens":0,"totalTokens":19}}}}' '{"method":"turn/completed","params":{"threadId":"thread-test","turn":{"id":"turn-test","status":"completed"}}}'
      ;;
  esac
done
"#,
                (12, 7),
            ),
            (
                LocalCliKind::Claude,
                r#"#!/bin/sh
if [ "$1" = '--help' ]; then
  printf '%s\n' '--json-schema --restricted --safe-mode --permission-prompts --no-session-persistence --include-partial-messages'
  exit 0
fi
cat >/dev/null
printf '%s\n' '{"type":"stream_event","event":{"delta":{"type":"text_delta","text":"claude"}}}' '{"type":"result","subtype":"success","is_error":false,"usage":{"input_tokens":6,"cache_creation_input_tokens":2,"cache_read_input_tokens":3,"output_tokens":4},"structured_output":{"text":"claude ok","tool_calls":[]}}'
"#,
                (11, 4),
            ),
            (
                LocalCliKind::Copilot,
                r#"#!/bin/sh
if [ "$1" = '--help' ]; then
  printf '%s\n' '--output-format --stream --available-tools --deny-tool --disable-builtin-mcps --no-ask-user --no-custom-instructions --disallow-temp-dir --no-auto-update'
  exit 0
fi
case " $* " in *' --available-tools=voff_no_native_tools '*) ;; *) exit 3 ;; esac
case " $* " in *' --deny-tool=shell '*) ;; *) exit 4 ;; esac
[ "$(cd "$COPILOT_HOME" && pwd -P)" = "$(pwd -P)" ] || exit 5
[ "$COPILOT_AUTO_UPDATE" = false ] || exit 6
grep -q 'validate against this JSON Schema' || exit 7
printf '%s\n' '{"type":"session.start","data":{}}' '{"type":"assistant.message_delta","data":{"deltaContent":"{\"text\""}}' '{"type":"assistant.message","data":{"model":"gpt-test","content":"Working on it.","toolRequests":[]}}' '{"type":"assistant.message","data":{"model":"gpt-test","content":"```json\n{\"text\":\"copilot ok\",\"tool_calls\":[]}\n```","toolRequests":[]}}' '{"type":"result","exitCode":0,"usage":{"premiumRequests":1}}'
"#,
                (0, 0),
            ),
        ] {
            let executable = directory.path().join(kind.id());
            fs::write(&executable, script).unwrap();
            let mut permissions = fs::metadata(&executable).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&executable, permissions).unwrap();
            let mut provider = LocalCliProvider::new(kind, directory.path()).unwrap();
            provider.executable = executable;
            let provider: std::sync::Arc<dyn ModelProvider> = std::sync::Arc::new(provider);
            let (events, result) = crate::contract::run(
                std::sync::Arc::clone(&provider),
                request(ToolChoice::Auto),
                CancellationToken::new(),
            )
            .await;
            crate::contract::assert_successful_stream(&events, &result.unwrap());
            assert!(events.iter().any(
                |event| matches!(event, ProviderEvent::TextDelta(text) if text.ends_with("ok"))
            ));
            let actual_usage = events
                .iter()
                .filter_map(|event| match event {
                    ProviderEvent::Usage(usage) => Some((usage.input_tokens, usage.output_tokens)),
                    _ => None,
                })
                .fold((0, 0), |(input, output), (next_input, next_output)| {
                    (input + next_input, output + next_output)
                });
            assert_eq!(actual_usage, expected_usage);

            let cancellation = CancellationToken::new();
            cancellation.cancel();
            let (events, result) =
                crate::contract::run(provider, request(ToolChoice::Auto), cancellation).await;
            crate::contract::assert_failed_stream(
                &events,
                &result.unwrap_err(),
                ProviderErrorKind::Cancelled,
            );
        }
    }
}
