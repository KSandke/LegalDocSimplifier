#!/usr/bin/env python3
"""
Test runner script for Legal Document Simplifier.

This script provides convenient ways to run different types of tests.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type=None, verbose=False, coverage=True, parallel=False):
    """Run tests with specified options."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    if test_type:
        if test_type == "classification":
            cmd.append("tests/classification/")
        elif test_type == "summarization":
            cmd.append("tests/summarization/")
        elif test_type == "simplification":
            cmd.append("tests/simplification/")
        elif test_type == "pipeline":
            cmd.append("tests/pipeline/")
        elif test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        else:
            print(f"Unknown test type: {test_type}")
            return False
    else:
        cmd.append("tests/")
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd) == 0


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="Run Legal Document Simplifier tests")
    parser.add_argument(
        "--type", 
        choices=["classification", "summarization", "simplification", "pipeline", "unit", "integration"],
        help="Type of tests to run"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--list", "-l", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    if args.list:
        # List available tests
        result = subprocess.run(["python", "-m", "pytest", "--collect-only", "tests/"], 
                              capture_output=True, text=True)
        print(result.stdout)
        return 0
    
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=not args.no_coverage,
        parallel=args.parallel
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
