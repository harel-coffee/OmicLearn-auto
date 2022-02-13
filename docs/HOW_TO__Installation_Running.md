## **Table of Contents**
- [Installation instructions](#installation-instructions)
- [Running](#running)
- [Running with Docker](#running-with-docker)

---

## Installation instructions

> We highly recommend the [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distribution which comes with a powerful package manager. 
>
> It is strongly recommended to install OmicLearn in its own environment.

1. Redirect to the folder of choice and clone the repository: `git clone https://github.com/OmicEra/OmicLearn`
2. Install the required packages with `conda env create --file environment.yml`
3. Activate the environment with  `conda activate omic_learn`

## Running

- To run OmicLearn, type the following command:

`streamlit run omic_learn.py --browser.gatherUsageStats False`

[Streamlit](https://www.streamlit.io/) gathers usage statistics per default, which we disable with this command.

> **Note:** A vanilla streamlit installation will show a menu bar in the upper left corner that offers additional functionality, such as recording screencasts. 
>
> For OmicLearn, this functionality is disabled. 

## Running with Docker

A docker instance should have at least 4 GB of memory. 
To build the docker, navigate to the OmicLearn directory: 

* `docker-compose build`

To run the docker container type:

* `docker-compose up`

* The OmicLearn page will be accessible via [`http://localhost:8501`](http://localhost:8501)