<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Person Re-Identification

Person Reidentification is an application that uses deep learning to accurately identify and re-identify individuals. The application will take the input from either a webcam or a pre-recorded video and run recognition against a database of each person that comes across the camera network. The application will identify the person in frame of a given camera and inform the user of each identified individual otherwise our software will create a new ID.

<!-- GETTING STARTED -->
### Installation
First make sure that python is installed: https://www.python.org/
Make sure to tick "Add to PATH" box

1. Create an empty folder to clone git
2. Open that folder in VScode
3. Open termimal (make sure it's a command prompt and not a powershell) and enter 'git clone https://github.com/Xaveee/deep-person-reid.git'
4. CLOSE the current folder and open the one you just clone (deep-person-reid)

## IMPORTANT: Make sure that you have deativated any conda environment. We dont want to install to the wrong environment

## Create virtual environment
5. Open the terminal (make sure it's a command prompt and not a powershell) and type 'py -m venv venv'
## Install packages
6. First activate the virtual environment by entering into the terminal '.\venv\scripts\activate'. You should see the '(venv)' at the beginning of your working folder
7. Change your python interpreter. Press 'Ctrl + Shift + P'. Look for 'Python: select interpreter'. Select 'enter interpreter path'. Select 'venv\scripts\python.exe'. Now you should see that you are using the venv interpreter in the bottom left corner
8. Install all packages by typing into the termimal 'pip install -r requirements.txt'
## Install torch. 
9. Type in the termimal 'pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html'
10. Type in the termimal 'python setup.py develop'

## Running the program
11. type in termimal 'cd ssd_mobilenet_v3' to change the working directory
12. To change the number of cameras in the camera network. Go to "ssd_mobilenet_v3\re-id.py" and change cam_num (line 15) value.
13. Use 'python re-id.py' to run. The time it takes to boot up depending on the processor of the computer. 


## Notes
Because of a bug in our program, in order to have the visualization saved, there must be a person present on the camera CONSECUTIVELY for at least 20 seconds.

To view the results, you can either use the .csv file labeled_gal.csv in "ssd_mobilenet_v3\data\labeled_gal.csv" or go to visualization folder in "ssd_mobilenet_v3\data\visualization". Each numbered folder is the label for each person identified.



### Running the program
11. type in termimal 'cd ssd_mobilenet_v3' to change the working directory
12. Use 'python re-id.py' to run
_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
