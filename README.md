
## Description

This project focuses on data analysis using Python, covering data preprocessing, visualization, and basic statistical analysis. The dataset used involves air quality measurements, and various Python libraries such as Pandas, Matplotlib, and Seaborn are utilized to analyze the data effectively.

## Installation

To set up the project, follow these steps:

1. Clone the repository (if you don have a file):
   ```bash
   git clone https://github.com/irfnriza/submission
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

4. Setup Environment

   - Shell/Terminal
      ```
      mkdir submission
      cd submission
      pipenv install
      pipenv shell
      pip install -r requirements.txt
      ```
   
   - for Anaconda
      ```
      conda create --name main-ds python=3.9
      conda activate main-ds
      pip install -r requirements.txt
      ```

4. Ensure Jupyter Notebook is installed (if using Jupyter):
   ```bash
   pip install jupyter
   ```

## Usage

To run the project, open Jupyter Notebook and execute the analysis scripts:
```bash
jupyter notebook
```

To run run steamlit app (dashboard)
```
streamlit run dashboard.py
```

Inside the notebook, you will find:
- **Data Preprocessing**: Handling missing values and cleaning the dataset.
- **Data Visualization**: Creating histograms, boxplots, and scatter plots.
- **Statistical Analysis**: Summarizing key insights from the dataset.

## Contributing

If you would like to contribute:
1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or collaboration opportunities, feel free to reach out:
- **Name**: Irfan rizadi
- **Email**: irfnriza@gmail.com
- **GitHub**: irfnriza

