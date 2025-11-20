use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use std::io;

/// Galaxy simulation with N-body physics
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
  /// Number of galaxies to simulate
  #[arg(short, long, default_value_t = 1)]
  galaxies: u32,
  /// Run in headless mode (no window)
  #[arg(long, default_value_t = false)]
  headless: bool,
  #[command(subcommand)]
  command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
  /// Generate shell completion scripts
  Completions {
    /// The shell to generate the script for
    #[arg(value_enum)]
    shell: Shell,
  },
}

fn main() {
  let args = Args::parse();

  if let Some(Commands::Completions { shell }) = args.command {
    let mut cmd = Args::command();
    let name = cmd.get_name().to_string();
    generate(shell, &mut cmd, name, &mut io::stdout());
    return;
  }

  galaxy_sim::state::run(args.galaxies, args.headless);
}
