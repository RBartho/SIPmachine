# SIP Machine

This project contains Python scripts to run a streamlit application "SIP machine" in your browser. This application can compute a number of commonly studied SIPs (statistical image properties) for aesthetic research.

# Installation instructions

Download all the files from this GitHub repository to your computer. (Download the ZIP file under the green "Code" button.) Then follow the installation instructions for your operating system:

[Linux Installation](docs/InstallationInstructions_Linux.md) \
[MacOS Installation](docs/InstallationInstructions_MacOS.md)  \
[Windows Installation](docs/InstallationInstructions_Windows.md) 

# Starting the application (after installation)

1. On MacOS and Linux open terminal, on Windows open Anaconda Prompt. Navigate to the downloaded folder containing the SIP_machine.py file.

2. Activate the created Python environment by typing into the terminal
```shell
conda activate SIP_machine
```
3. Now start the streamlit application from the terminal in the same folder as above with

```shell
python -m streamlit run SIP_machine.py
 ```

# Notes on using the application

1. If you want to restart the app, just refresh your browser. All loaded data will be removed and all active calculations will stop.

2. Dont interact with the application (e.g. selecting SIPs, Sidebar, uploading or deleting images) while SIP-computations are running. It will refresh the application and all progress will be lost.

3. Multithreading is not supported as it would limit platform independence. To calculate SIPs for very large datasets, you may want to consider splitting the data and running multiple instances of the application.

4. The number of images you can load into the application at one time is limited by the amount of RAM your computer has. Also, large images require much more processing time than smaller images.

# Privacy and security
All calculations and data transfers of the application take place on your local computer. The browser is only used as an interface. No data is uploaded to the Internet.
