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
  /// Particles per galaxy (overrides default)
  #[arg(long)]
  particles: Option<u32>,
  /// Save a PNG snapshot every N steps in headless mode (0 = off)
  #[arg(long, default_value_t = 0)]
  snapshot_every: u64,
  /// Write a particle CSV every N steps in headless mode (0 = off)
  #[arg(long, default_value_t = 0)]
  dump_every: u64,
  /// Stop headless mode after N steps (0 = run until Ctrl+C)
  #[arg(long, default_value_t = 0)]
  max_steps: u64,
  /// Directory for snapshots and dumps
  #[arg(long, default_value = "snapshots")]
  out_dir: String,
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

  galaxy_sim::state::run(galaxy_sim::RunConfig {
    galaxies: args.galaxies,
    headless: args.headless,
    particles: args.particles,
    snapshot_every: args.snapshot_every,
    dump_every: args.dump_every,
    max_steps: args.max_steps,
    out_dir: args.out_dir,
  });
}
