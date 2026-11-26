//! A throw in one DSO must be caught as `linkcell::Error` in another, and
//! the status and message must survive. libc++ matches a handler by the
//! typeinfo address, so Linux and macOS also require one address. MSVC
//! matches the decorated name and gives each module its own `type_info`,
//! so address equality is not the invariant there. The owner is one shared
//! library. Windows uses `dllexport` in that library and `dllimport` in
//! consumers. A fully static link is a separate test: one copy in the final
//! image.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn command_line(cmd: &Command) -> String {
    let mut parts = Vec::new();
    parts.push(cmd.get_program().to_string_lossy().into_owned());
    for arg in cmd.get_args() {
        parts.push(arg.to_string_lossy().into_owned());
    }
    parts.join(" ")
}

fn run(cmd: &mut Command) -> std::process::Output {
    let shown = command_line(cmd);
    let output = cmd
        .output()
        .unwrap_or_else(|err| panic!("failed to start {shown}: {err}"));
    if !output.status.success() {
        panic!(
            "command failed ({})\n{shown}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    output
}

fn host_triple() -> String {
    let output = Command::new("rustc")
        .args(["-vV"])
        .output()
        .expect("rustc -vV");
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("rustc -vV host")
        .to_string()
}

fn cxx_tool() -> cc::Tool {
    let host = host_triple();
    // cc::Build reads Cargo build-script variables. This integration test
    // does not have them. An explicit host that is not Cargo's TARGET makes
    // the probe use `rustc --print=cfg` instead.
    static CLEAR_CARGO_BUILD_ENV: std::sync::Once = std::sync::Once::new();
    CLEAR_CARGO_BUILD_ENV.call_once(|| {
        std::env::remove_var("TARGET");
        std::env::remove_var("HOST");
        std::env::remove_var("OPT_LEVEL");
    });
    cc::Build::new()
        .cargo_metadata(false)
        .cpp(true)
        .std("c++17")
        .opt_level(0)
        .debug(false)
        .host(&host)
        .target(&host)
        .get_compiler()
}

fn lib_file(stem: &str, msvc: bool) -> String {
    if msvc {
        format!("{stem}.dll")
    } else if cfg!(windows) {
        format!("lib{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

fn exe_file(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

fn rpath(dir: &Path) -> String {
    format!("-Wl,-rpath,{}", dir.display())
}

struct Symbols {
    undefined: bool,
    defined: bool,
}

fn classify(nm_text: &str, symbol: &str) -> Symbols {
    let mut undefined = false;
    let mut defined = false;
    for line in nm_text.lines() {
        if !line.contains(symbol) || line.contains("typeinfo name for") {
            continue;
        }
        let kind = line.split_whitespace().next().unwrap_or("");
        if kind == "U" {
            undefined = true;
        } else {
            defined = true;
        }
    }
    Symbols { undefined, defined }
}

fn assert_undefined(nm_text: &str, symbol: &str) {
    let found = classify(nm_text, symbol);
    assert!(
        found.undefined && !found.defined,
        "DSO should reference {symbol} and not define it\n{nm_text}"
    );
}

fn assert_defined_once(nm_text: &str, symbol: &str) {
    let mut defined = 0;
    for line in nm_text.lines() {
        if !line.contains(symbol) || line.contains("typeinfo name for") {
            continue;
        }
        let kind = line.split_whitespace().next().unwrap_or("");
        if kind != "U" {
            defined += 1;
        }
    }
    assert_eq!(defined, 1, "expected one definition of {symbol}\n{nm_text}");
}

fn nm_text(path: &Path) -> String {
    let output = run(Command::new("nm").args(["-C"]).arg(path));
    String::from_utf8_lossy(&output.stdout).into_owned()
}

struct Build {
    tool: cc::Tool,
    msvc: bool,
    root: PathBuf,
    include: PathBuf,
    dir: PathBuf,
}

impl Build {
    fn new(dir: PathBuf) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tool = cxx_tool();
        let msvc = tool.is_like_msvc();
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        Self {
            tool,
            msvc,
            include: root.join("include"),
            root,
            dir,
        }
    }

    fn compile(&self, args: &[OsString]) {
        let mut cmd = self.tool.to_command();
        cmd.current_dir(&self.dir);
        if self.msvc {
            cmd.arg("-EHsc");
        }
        cmd.args(args);
        run(&mut cmd);
    }

    fn shared_owner(&self) -> PathBuf {
        let out = self.dir.join(lib_file("linkcell_eh", self.msvc));
        let source = self.root.join("src/error.cpp");
        if self.msvc {
            let obj = self.dir.join("error.obj");
            self.compile(&[
                OsString::from("-DLINKCELL_EH_BUILDING"),
                OsString::from("-I"),
                self.include.as_os_str().to_owned(),
                OsString::from("-LD"),
                OsString::from(format!("-Fo{}", obj.display())),
                OsString::from(format!("-Fe{}", out.display())),
                source.as_os_str().to_owned(),
            ]);
        } else {
            self.compile(&[
                OsString::from("-shared"),
                OsString::from("-DLINKCELL_EH_BUILDING"),
                OsString::from("-I"),
                self.include.as_os_str().to_owned(),
                source.as_os_str().to_owned(),
                OsString::from("-o"),
                out.as_os_str().to_owned(),
            ]);
        }
        out
    }

    fn shared_hidden(&self, tag: &str, define_b: bool) -> PathBuf {
        let out = self.dir.join(lib_file(tag, self.msvc));
        let source = self.root.join("tests/exception_typeinfo_hidden.cpp");
        if self.msvc {
            let obj = self.dir.join(format!("{tag}.obj"));
            let mut args = vec![
                OsString::from("-I"),
                self.include.as_os_str().to_owned(),
                OsString::from("-LD"),
                OsString::from(format!("-Fo{}", obj.display())),
                OsString::from(format!("-Fe{}", out.display())),
            ];
            if define_b {
                args.push(OsString::from("-DLC_PROBE_B"));
            }
            args.push(source.as_os_str().to_owned());
            args.push(OsString::from("-link"));
            args.push(self.dir.join("linkcell_eh.lib").into_os_string());
            self.compile(&args);
        } else {
            let mut args = vec![
                OsString::from("-shared"),
                OsString::from("-fvisibility=hidden"),
                OsString::from("-fvisibility-inlines-hidden"),
                OsString::from("-I"),
                self.include.as_os_str().to_owned(),
            ];
            if define_b {
                args.push(OsString::from("-DLC_PROBE_B"));
            }
            args.push(source.as_os_str().to_owned());
            args.push(OsString::from("-L"));
            args.push(self.dir.as_os_str().to_owned());
            args.push(OsString::from("-llinkcell_eh"));
            args.push(OsString::from(rpath(&self.dir)));
            args.push(OsString::from("-o"));
            args.push(out.as_os_str().to_owned());
            self.compile(&args);
        }
        out
    }

    fn probe(&self, hidden_a: &str, hidden_b: &str) -> PathBuf {
        let out = self.dir.join(exe_file("exception_typeinfo"));
        let source = self.root.join("tests/exception_typeinfo.cpp");
        if self.msvc {
            let obj = self.dir.join("probe.obj");
            self.compile(&[
                OsString::from("-I"),
                self.include.as_os_str().to_owned(),
                OsString::from(format!("-Fo{}", obj.display())),
                OsString::from(format!("-Fe{}", out.display())),
                source.as_os_str().to_owned(),
                OsString::from("-link"),
                self.dir.join(format!("{hidden_a}.lib")).into_os_string(),
                self.dir.join(format!("{hidden_b}.lib")).into_os_string(),
                self.dir.join("linkcell_eh.lib").into_os_string(),
            ]);
        } else {
            self.compile(&[
                OsString::from("-I"),
                self.include.as_os_str().to_owned(),
                source.as_os_str().to_owned(),
                OsString::from("-L"),
                self.dir.as_os_str().to_owned(),
                OsString::from(format!("-l{hidden_a}")),
                OsString::from(format!("-l{hidden_b}")),
                OsString::from("-llinkcell_eh"),
                OsString::from(rpath(&self.dir)),
                OsString::from("-o"),
                out.as_os_str().to_owned(),
            ]);
        }
        out
    }

    fn run_probe(&self, probe: &Path) {
        let mut cmd = Command::new(probe);
        if cfg!(windows) {
            let mut path = OsString::from(&self.dir);
            if let Some(old) = std::env::var_os("PATH") {
                path.push(";");
                path.push(old);
            }
            cmd.env("PATH", path);
        } else {
            cmd.env("LD_LIBRARY_PATH", &self.dir);
            cmd.env("DYLD_LIBRARY_PATH", &self.dir);
        }
        run(&mut cmd);
    }
}

#[test]
fn hidden_dso_shares_error_typeinfo() {
    let dir = std::env::temp_dir().join(format!(
        "linkcell-eh-typeinfo-{}-shared",
        std::process::id()
    ));
    let build = Build::new(dir);
    let owner = build.shared_owner();
    let hidden_a = build.shared_hidden("exception_typeinfo_hidden_a", false);
    let hidden_b = build.shared_hidden("exception_typeinfo_hidden_b", true);
    let probe = build.probe("exception_typeinfo_hidden_a", "exception_typeinfo_hidden_b");

    if cfg!(target_os = "linux") {
        let owner_nm = nm_text(&owner);
        assert_defined_once(&owner_nm, "typeinfo for linkcell::Error");
        assert!(
            classify(&owner_nm, "linkcell::Error::~Error()").defined,
            "owner should define the key function\n{owner_nm}"
        );
        for hidden in [&hidden_a, &hidden_b] {
            let text = nm_text(hidden);
            assert_undefined(&text, "typeinfo for linkcell::Error");
            assert_undefined(&text, "linkcell::Error::~Error()");
        }
    }

    build.run_probe(&probe);
}

#[test]
fn static_link_has_one_error_typeinfo() {
    let dir = std::env::temp_dir().join(format!(
        "linkcell-eh-typeinfo-{}-static",
        std::process::id()
    ));
    let build = Build::new(dir);
    let out = build.dir.join(exe_file("exception_typeinfo_static"));
    let main = build.root.join("tests/exception_typeinfo_static.cpp");
    let other = build.root.join("tests/exception_typeinfo_static_other.cpp");
    let source = build.root.join("src/error.cpp");
    if build.msvc {
        build.compile(&[
            OsString::from("-DLINKCELL_EH_STATIC"),
            OsString::from("-I"),
            build.include.as_os_str().to_owned(),
            OsString::from(format!("-Fe{}", out.display())),
            main.as_os_str().to_owned(),
            other.as_os_str().to_owned(),
            source.as_os_str().to_owned(),
        ]);
    } else {
        build.compile(&[
            OsString::from("-DLINKCELL_EH_STATIC"),
            OsString::from("-I"),
            build.include.as_os_str().to_owned(),
            main.as_os_str().to_owned(),
            other.as_os_str().to_owned(),
            source.as_os_str().to_owned(),
            OsString::from("-o"),
            out.as_os_str().to_owned(),
        ]);
    }
    if cfg!(target_os = "linux") {
        let text = nm_text(&out);
        assert_defined_once(&text, "typeinfo for linkcell::Error");
    }
    run(&mut Command::new(&out));
}
