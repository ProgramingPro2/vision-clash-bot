# vision-clash-bot

**vision-clash-bot** is an enhanced fork of py-clash-bot that adds cutting-edge AI and computer vision capabilities to Clash Royale automation. This advanced bot uses movement-based unit tracking, Deep Q-Learning neural networks, and real-time OCR to make intelligent gameplay decisions, revolutionizing how automation interacts with Clash Royale.

## 🧠 **NEW: AI-Powered Features**

### **Movement-Based Unit Tracking**

- **Real-time unit detection** using negative greyscale frame differencing
- **Persistent unit IDs** with centroid-based tracking across frames
- **Speed vector calculation** and direction analysis
- **Occlusion handling** for robust tracking during battles

### **Deep Q-Learning AI**

- **Neural network decision making** for optimal card play strategies
- **Experience replay system** for continuous learning
- **State representation** including unit positions, speeds, and game context
- **Reward-based learning** from battle outcomes

### **Advanced Computer Vision**

- **OCR-based tower health detection** with physical screen measurements
- **Health bar presence detection** (only appears after damage)
- **Real-time visualization** with bounding boxes and unit IDs
- **Enhanced GUI** with movement bot controls and training metrics

## 🎯 **Enhanced Automation**

**vision-clash-bot** builds upon the original py-clash-bot with these revolutionary improvements:

- **Intelligent card placement** based on unit movement patterns
- **Adaptive strategies** that learn from each battle
- **Real-time unit tracking** at 30+ FPS processing
- **Modular architecture** for easy testing and development

## ✨ Features

### 🎮 **Battle Automation**

- **Trophy Road 1v1 Battles** - Automatically fight in trophy road ladder matches
- **Path of Legends 1v1 Battles** - Battle in the competitive Path of Legends mode
- **2v2 Battles** - Team up with clan members for 2v2 matches
- **Random Decks** - Randomize your deck selection before each battle
- **Smart Battle Management** - Skip fights when chests are full, disable win/loss tracking

### 🎁 **Rewards & Collection**

- **Card Mastery Rewards** - Collect mastery rewards earned from battles
- **Card Upgrades** - Upgrade your current deck after each battle

### ⚙️ **Advanced Settings**

- **Emulator Support** - Works with both MEmu and Google Play Games emulators
- **Render Mode Selection** - Choose between OpenGL, DirectX, and Vulkan rendering
- **Real-time Statistics** - Track wins, losses, chests opened, and more
- **Performance Monitoring** - Monitor bot runtime, failures, and account switches

## 📥 **Download & Installation**

### **Option 1: Pre-built Windows Executable (Recommended)**

The latest Windows executable is automatically built using GitHub Actions:

1. **Go to Actions**: Visit [https://github.com/ProgramingPro2/vision-clash-bot/actions](https://github.com/ProgramingPro2/vision-clash-bot/actions)
2. **Select Latest Build**: Click on the most recent successful workflow run
3. **Download Artifacts**: Scroll down to "Artifacts" section
4. **Download `windows-build`**: This contains both the executable and MSI installer
5. **Extract and Run**: Extract the zip file and run `py-clash-bot.exe`

**Build Contents:**

- **`py-clash-bot.exe`** - Standalone executable (5-6GB with all dependencies)
- **`py-clash-bot-0.0.0-amd64.msi`** - Windows installer for easy setup
- **All AI models and dependencies** included

### **Option 2: Run from Source (Development/Testing)**

For developers or users who want to run the latest code directly:

#### **Prerequisites**

- **Python 3.12** (required)
- **Git** (for cloning the repository)
- **Windows 10/11** (64-bit)
- **8GB RAM minimum** (16GB recommended for AI features)

#### **Installation Steps**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ProgramingPro2/vision-clash-bot.git
   cd vision-clash-bot
   ```

2. **Create and activate virtual environment**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Upgrade pip**:

   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install the project in editable mode**:

   ```bash
   pip install -e .
   ```

5. **Run the application**:
   ```bash
   python -m pyclashbot
   ```

#### **Alternative: Using uv (Faster)**

If you have `uv` installed (recommended for faster dependency management):

```bash
# Install uv if you don't have it
pip install uv

# Install dependencies using uv
uv sync

# Activate the virtual environment
.venv\Scripts\activate

# Run the application
python -m pyclashbot
```

#### **Development Benefits**

- **Immediate code changes** - No need to rebuild after modifications
- **Debugging support** - Full Python debugging capabilities
- **Latest features** - Access to the most recent code changes
- **Custom modifications** - Easy to modify and test new features

### **System Requirements**

- **Windows 10/11** (64-bit)
- **8GB RAM minimum** (16GB recommended for AI features)
- **2GB free disk space**
- **Android emulator** (MEmu or Google Play Games)

## 🚀 Setup Instructions

**vision-clash-bot** supports two emulators. Choose the one that works best for your system:

### Option 1: MEmu Emulator

1. **Download MEmu 9.2.5.0** - Get it from the [official site](https://www.memuplay.com/) or use this [working installer](https://drive.google.com/file/d/1FDMa5oKIhbM_X2TGHg6qSi3bnIuIXcPf/view?usp=sharing) (version 9.2.5.0 recommended)
2. **Install MEmu** - Run the MEmu installer
3. **Download vision-clash-bot** - Get the latest build from [GitHub Actions](https://github.com/ProgramingPro2/vision-clash-bot/actions) (see Download section above)
4. **Install py-clash-bot** - Run the installer
5. **Create the VM** - Start the bot once to let it automatically create the "pyclashbot-96" MEmu emulator
6. **Install Clash Royale** - Install Clash Royale manually on the "pyclashbot-96" emulator via Google Play Store
7. **Complete setup** - Open Clash Royale manually, complete the tutorial, and optionally sign in to your account
8. **Close MEmu** - Close the MEmu emulator completely
9. **Start automation** - Start the bot, configure your settings, then click "Start" to begin automation

**Troubleshooting MEmu:**

- Switch render mode to Vulkan, DirectX, or OpenGL if experiencing issues
- Delete the VM and let the bot create a new one
- Enable UEFI in BIOS if needed

### Option 2: Google Play Games Emulator

1. **Download Google Play Games Emulator** - Get it from [https://developer.android.com/games/playgames/emulator](https://developer.android.com/games/playgames/emulator)
2. **Install the emulator** - Run the Google Play installer
3. **Initial setup** - Boot the Google Play Games Emulator once. This will trigger a Google sign-in flow in your web browser - complete this process. If prompted to allow USB debugging, click "Accept"
4. **Download vision-clash-bot** - Get the latest build from [GitHub Actions](https://github.com/ProgramingPro2/vision-clash-bot/actions) (see Download section above)
5. **Install Clash Royale** - Download Clash Royale manually from the emulator
6. **Complete setup** - Start Clash Royale manually, complete the tutorial, and optionally sign in to your account
7. **Optional: Set display ratio** - Go to Google Play Emulator > Developer Options > Display Ratio > 9:16 (Portrait) for optimal look
8. **Close emulator** - Close the Google Play emulator completely
9. **Start automation** - Start the bot, configure your settings, then click "Start" to begin automation

## 🤖 **Movement Bot Features**

### **New GUI Tabs**

- **Movement Bot Tab** - Access AI controls and training settings
- **Model Management** - Delete current model, reset training
- **Real-time Visualization** - See tracked units with bounding boxes and IDs
- **Training Metrics** - Monitor DQN learning progress and performance

### **AI Configuration**

- **Enable/Disable Movement Detection** - Toggle the new AI system
- **Training Mode** - Let the bot learn from battles
- **Model Save/Load** - Persist learned strategies
- **Performance Monitoring** - Track AI decision accuracy

### **Advanced Settings**

- **Unit Tracking Sensitivity** - Adjust detection parameters
- **OCR Health Detection** - Configure tower health monitoring
- **Speed Vector Visualization** - See unit movement patterns
- **Training Data Collection** - Gather battle experience for learning

### Important Notes

- **Language Setting** - Ensure Clash Royale is set to English for optimal bot performance
- **Tutorial Completion** - The tutorial must be completed manually before starting the bot
- **Account Setup** - Sign in with SuperCell ID or create a new account as needed

## 🔧 Emulator Debugging

Having trouble with your emulator? This section provides troubleshooting tips for common issues with both supported emulators.

### Google Play Games Emulator Debugging

- **Use the correct version** - Make sure you're using the DEVELOPER Google Play Games emulator, not the BETA version. Download it from [https://developer.android.com/games/playgames/emulator](https://developer.android.com/games/playgames/emulator)
- **Watch for login prompts** - Google Play makes a popup in your default browser for the Google sign-in prompt. Sometimes you might miss this during emulator boot, and it'll hang forever. If you're experiencing booting issues, check for a login prompt in a minimized browser window!
- **Adjust rendering settings** - If it's still not rendering properly, try adjusting render mode settings at System tray > Google Play Games emulator > Graphics settings > Vulkan device override OR Graphics > Graphics stack override

### MEmu Emulator Debugging

- **Hardware requirements** - MEmu is more hardware intensive, so if you're on a low-end machine try using Google Play Games emulator instead
- **Black screen or boot issues** - If it's showing a black screen or never fully booting, try adjusting render mode via the ClashBot settings, then start the bot to apply those settings
- **BIOS requirements** - MEmu REQUIRES your BIOS to have UEFI and Hyper-V enabled!
  - Enable UEFI: [https://www.youtube.com/watch?v=uAMLGIlFMdI](https://www.youtube.com/watch?v=uAMLGIlFMdI)
  - Enable Hyper-V: [https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/install-hyper-v?tabs=powershell&pivots=windows](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/install-hyper-v?tabs=powershell&pivots=windows)
- **Version conflicts** - Some old versions of pyclashbot create corrupt instances of MEmu. If you're switching between versions and MEmu is breaking, try deleting your existing MEmu VMs, or reinstalling MEmu entirely

## 🎯 Demo

<img src="https://github.com/pyclashbot/py-clash-bot/blob/master/assets/demo-game.gif?raw=true" width="50%" alt="Game Demo"/><img src="https://github.com/pyclashbot/py-clash-bot/blob/master/assets/demo-gui.gif?raw=true" width="50%" alt="GUI Demo"/>

_Left: Bot automation in action | Right: User interface and controls_

## 🤝 Contributing

We welcome contributions from the community! Whether you have ideas for new features, bug reports, or want to help with development, there are many ways to get involved:

- **Report Issues** - Open an issue on [GitHub Issues](https://github.com/pyclashbot/py-clash-bot/issues)
- **Feature Requests** - Suggest new automation features or improvements
- **Code Contributions** - Check out our [Contributing Guide](CONTRIBUTING.md)
- **Community Support** - Help other users on our [Discord server](https://discord.gg/nqKRkyq2UU)

## ⚠️ Disclaimer

This tool is designed for educational and automation purposes. Please ensure you comply with Clash Royale's Terms of Service and use responsibly. The developers are not responsible for any consequences resulting from the use of this software.

---

**Made with ❤️ by the py-clash-bot community**

_Automate your Clash Royale experience and focus on what matters most - strategy and fun!_
