# create and run virtual environment

"""
python3 -m venv .venv
source .venv/bin/activate
"""

# deactivate virtual environment

"""
deactivate
"""

select kernal > select another kernal > sleect .venv

# verify everything is pointing to the correct environment:

"""
which python
which pip
echo $VIRTUAL_ENV
"""

# install required libraries

pip3 install numpy pandas seaborn matplotlib scikit-learn scipy scikit-learn

pip3 freeze > requirements.txt

pip3 install -r requirements.txt

streamlit run app.py
