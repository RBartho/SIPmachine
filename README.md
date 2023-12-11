# SIP Machine

This project contains Python scripts to run a streamlit application in your browser. This application computes a number of commonly studied SIPs (statistical image properties) for aesthetic research.

# Installation instructions
[Linux Installation](docs/InstallationInstructions_Linux.md) \
[Linux Installation](docs/InstallationInstructions_MacOS.md)  \
[Linux Installation](docs/InstallationInstructions_Windows.md) 

# Starting the application (after installation)

1. Open a terminal on your system and navigate to the downloaded folder containing the SIP_machine.py file.

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

3. Multithreading is not supported. To calculate SIPs for very large datasets, you may want to consider running multiple instances of the application.

# Privacy and security
All calculations and data transfers take place on the local computer. The browser is only used as an interface. No data is uploaded to the Internet.
