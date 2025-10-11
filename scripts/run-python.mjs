#!/usr/bin/env node

/**
 * Lightweight Python launcher for pnpm scripts.
 *
 * Usage examples:
 *   pnpm run py backtester/backtest_m001.py -- --quick
 *   pnpm run py -m pip install -r requirements.txt
 *
 * The script selects an available Python interpreter in the following order:
 *   1. Active Conda environment (CONDA_PYTHON_EXE or CONDA_PREFIX)
 *   2. process.env.PYTHON
 *   3. process.env.PYTHON_BIN (legacy support)
 *   4. "python3"
 *   5. "python"
 */

import { spawn, spawnSync } from "node:child_process";
import { join } from "node:path";
import process from "node:process";

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("Usage: pnpm run py <python-args...>");
  process.exit(1);
}

const candidates = [];

const condaPrefix = process.env.CONDA_PREFIX;
if (condaPrefix) {
  const condaCmd =
    process.platform === "win32"
      ? join(condaPrefix, "python.exe")
      : join(condaPrefix, "bin", "python");
  candidates.push(condaCmd);
}

const condaPython = process.env.CONDA_PYTHON_EXE;
if (condaPython && condaPython.startsWith(condaPrefix || "")) {
  candidates.push(condaPython);
}

candidates.push(
  process.env.PYTHON,
  process.env.PYTHON_BIN,
  "python3",
  "python"
);

const DEFAULT_CONDA_ENV = "/home/snowypainter/miniconda3/envs/research";
if (process.platform !== "win32") {
  candidates.push(join(DEFAULT_CONDA_ENV, "bin", "python"));
} else {
  candidates.push(join(DEFAULT_CONDA_ENV, "python.exe"));
}

const uniqueCandidates = [...new Set(candidates.filter(Boolean))];

let pythonCmd = null;
console.log("Trying python interpreters:", uniqueCandidates);
for (const cmd of uniqueCandidates) {
  const probe = spawnSync(cmd, ["--version"], {
    stdio: "ignore",
  });
  if (!probe.error && probe.status === 0) {
    pythonCmd = cmd;
    console.log("Selected python:", pythonCmd);
    break;
  }
}

if (!pythonCmd) {
  console.error(
    "❌ Python interpreter not found. Activate your Conda/virtual environment or set the PYTHON environment variable."
  );
  process.exit(1);
}

const env = { ...process.env };
const pathSep = process.platform === "win32" ? ";" : ":";
const projectRoot = process.cwd();
env.PYTHONPATH = env.PYTHONPATH
  ? `${projectRoot}${pathSep}${env.PYTHONPATH}`
  : projectRoot;
console.log("Project root:", projectRoot);
console.log("PYTHONPATH:", env.PYTHONPATH);

const child = spawn(pythonCmd, args, {
  stdio: "inherit",
  env,
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 1);
  }
});
