# Set up your development environment

## Prerequisites

### Toolchain

We use the following tools to build and test the code:

#### NodeJS

We high recommended using [volta](https://volta.sh/) to manage your nodejs environment.

Volta better than nvm because it could automatically detect the Node.js version required by our project.

```bash
curl https://get.volta.sh | bash
volta install node@18
```

#### Rust

Rust is a required dependency for building some native gems. And, [rustup](https://www.rust-lang.org/tools/install) is a recommended way to install rust toolchain.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Python

We use [pdm](https://pdm.fming.dev/) to manage python dependencies.

```bash
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -
```

#### VS Code

Sure you can use any other editor/IDE you like, but we highly recommend [VSCode](https://code.visualstudio.com/) for a better out-of-the-box experience.