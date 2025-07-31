#!/usr/bin/env python3
"""
Development utility script for the Financial Time Series Analysis Platform.
Provides easy commands to manage the Docker development environment.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command"""
    print(f"Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result
    else:
        result = subprocess.run(cmd, shell=True, check=check)
        return result

def docker_compose(*args):
    """Run docker compose command"""
    cmd = f"docker compose {' '.join(args)}"
    return run_command(cmd)

def check_docker():
    """Check if Docker is running"""
    try:
        run_command("docker info", capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Docker is not running. Please start Docker Desktop.")
        return False

def setup_env():
    """Set up environment file"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_file.exists() and env_example.exists():
        print("[INFO] Creating .env file from .env.example...")
        env_content = env_example.read_text()
        env_file.write_text(env_content)
        print("[OK] Created .env file. Edit it with your configuration if needed.")
    elif env_file.exists():
        print("[OK] .env file already exists.")
    else:
        print("[WARN] No .env.example file found.")

def start():
    """Start the development environment"""
    if not check_docker():
        return
    
    setup_env()
    
    print("[START] Starting Financial Platform development environment...")
    docker_compose("up", "-d", "--build")
    
    print("\n[WAIT] Waiting for services to be ready...")
    time.sleep(5)
    
    # Check service health
    print("\n[CHECK] Checking service status...")
    docker_compose("ps")
    
    print("\n[SUCCESS] Development environment started!")
    print("\n[SERVICES] Available services:")
    print("   * Frontend App: http://localhost:3000")
    print("   * Backend API: http://localhost:8000")
    print("   * API Docs: http://localhost:8000/docs")
    print("   * PostgreSQL: localhost:5432")
    print("   * Redis: localhost:6379")
    
    print("\n[INFO] To view logs: python dev.py logs")
    print("[INFO] To stop: python dev.py stop")

def stop():
    """Stop the development environment"""
    print("[STOP] Stopping Financial Platform development environment...")
    docker_compose("down")
    print("[OK] Development environment stopped!")

def restart():
    """Restart the development environment"""
    stop()
    start()

def logs(*services):
    """Show logs for services"""
    if services:
        docker_compose("logs", "-f", *services)
    else:
        docker_compose("logs", "-f")

def shell(service="backend"):
    """Open shell in a service container"""
    print(f"[SHELL] Opening shell in {service} container...")
    docker_compose("exec", service, "/bin/bash")

def test():
    """Run tests in the backend container"""
    print("[TEST] Running tests...")
    docker_compose("exec", "backend", "python", "-m", "pytest", "-v")

def status():
    """Show status of all services"""
    print("[STATUS] Service Status:")
    docker_compose("ps")
    
    print("\n[HEALTH] Health Checks:")
    # Check backend health
    try:
        result = run_command("curl -s http://localhost:8000/health", capture_output=True)
        if result.returncode == 0:
            print("   * Backend API: [OK] Healthy")
        else:
            print("   * Backend API: [ERROR] Unhealthy")
    except:
        print("   * Backend API: [ERROR] Not responding")

def clean():
    """Clean up Docker resources"""
    print("[CLEAN] Cleaning up Docker resources...")
    docker_compose("down", "-v", "--remove-orphans")
    run_command("docker system prune -f")
    print("[OK] Cleanup complete!")

def build():
    """Build Docker images"""
    print("[BUILD] Building Docker images...")
    docker_compose("build", "--no-cache")
    print("[OK] Build complete!")

def help_msg():
    """Show help message"""
    print("""
Financial Time Series Analysis Platform - Development Tool

Usage: python dev.py <command>

Available commands:
    start      Start the development environment
    stop       Stop the development environment  
    restart    Restart the development environment
    logs       Show logs for all services (or specify service names)
    shell      Open shell in backend container (or specify service name)
    test       Run tests in the backend container
    status     Show status of all services
    clean      Clean up Docker resources
    build      Build Docker images
    help       Show this help message

Examples:
    python dev.py start
    python dev.py logs backend
    python dev.py shell postgres
    python dev.py test
""")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        help_msg()
        return
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    commands = {
        'start': start,
        'stop': stop,
        'restart': restart,
        'logs': lambda: logs(*args),
        'shell': lambda: shell(args[0] if args else "backend"),
        'test': test,
        'status': status,
        'clean': clean,
        'build': build,
        'help': help_msg,
    }
    
    if command in commands:
        try:
            commands[command]()
        except KeyboardInterrupt:
            print("\n\n[EXIT] Goodbye!")
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            sys.exit(1)
    else:
        print(f"[ERROR] Unknown command: {command}")
        help_msg()
        sys.exit(1)

if __name__ == "__main__":
    main()