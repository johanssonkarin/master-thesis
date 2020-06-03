# Master Thesis Project
Energy consumption model for the local grid level which is a part of a Master Thesis Project in Sociotechnical Systems Engineering at Uppsala University. This is just a short overview of the repository. For full background and extensive explaintions of the model please see the thesis. The model flow could be described somewhat like the illustation below:

<img src="https://github.com/johanssonkarin/master-thesis/blob/master/modellflownew.png" width="400" height="auto" align="middle" />


## Repository navigation

### code
Contains the different code-files which make up the model. At the top level general GUI file is found and then the folder is further devided into subfolders:

* dev - A mix of code files and snippets which were kept throughout the development phase.
* files - Simply the different model and class files.

### data
Due to the limited amount of data, the model is currently only developed for the Stockholm region. The model can be expanded to include other regions by adding another folder, carrying the same data structure as the Stockholm folder, to favor scalability. The folders ‘Residential’, ‘Office’ and ‘PV’ all include several csv-files which are named according to the classes or attributes to which the particular data corresponds to. The ‘EV’ folder contains transition matrices and distance data, for weekdays and weekends, which are used within the class for EV charging stations. 

<img src="https://github.com/johanssonkarin/master-thesis/blob/master/folder_structure.png" width="250" height="auto" align="middle" />


## Built With

* [Python](https://www.python.org/) - Used for the object-oriented power grid structure.
* [Jupyter Notebooks](https://jupyter.org/) - Used for code simulation and visualization.
* [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/index.html) - Used for creating the GUI.
* [EV Spatial Model](https://sheperomah.github.io/EVSpatialChargingModel/build/index.html#) - Used for simulations of EVs within the model.


## Author

* **Karin Johansson** - [johanssonkarin](https://github.com/johanssonkarin)

## License

This project is not yet licensed since it is not sure what will happen to the code after the project is finished. If you happen to see this repo - please leave it be 🙏

## Some GUI Screenshots

The input tabs...


<img src="https://github.com/johanssonkarin/master-thesis/blob/master/GUIinput1.png" width="600" height="auto" align="middle" />

<img src="https://github.com/johanssonkarin/master-thesis/blob/master/GUIinput2.png" width="600" height="auto" align="middle" />


... and the output tabs.


<img src="https://github.com/johanssonkarin/master-thesis/blob/master/GUIoutput1.png" width="600" height="auto" align="middle" />

<img src="https://github.com/johanssonkarin/master-thesis/blob/master/GUIoutput2.png" width="600" height="auto" align="middle" />
