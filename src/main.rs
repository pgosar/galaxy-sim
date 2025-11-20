use clap::Parser;

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
}

fn main() {
  let args = Args::parse();
  galaxy_sim::state::run(args.galaxies, args.headless);
}
