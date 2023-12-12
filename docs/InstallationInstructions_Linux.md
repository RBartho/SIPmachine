# Installation instructions Linux


1. Download all files from this GitHub repository to your computer. (Download under green "Code" Button.)

2. If you do not already have Anaconda or Miniconda installed, download and install Anaconda:

	https://www.anaconda.com/download

3. Open a terminal on your system. If you do not know how to do this on your system, google it :-).

4. Navigate to the downloaded files in the terminal window. If you do not know how to change folders in terminal, google it :-). 

5. In the same folder where the file "requirements.txt" is, run this command in the terminal:

```shell
conda create --name SIP_machine -y
```

This should create a python enviroment with the name "SIPmachine".  


6. Activate the new environment by typing into the terminal:

```shell
conda activate SIP_machine
```

7. Install all needed python packages into the new python enviroment by:

```shell
conda install --file requirements.txt -y
```
	
9. Now launch the streamlit application from the terminal in the same folder as above:

```shell
python -m streamlit run SIP_machine.py
```



Your default browser should open the application on your local machine. It should look like this: 
![Screenshot](https://github.com/RBartho/SIPmachine/master/toolbox_screenshot.png)
The browser is only used as an interface. No data is uploaded to the Internet.

