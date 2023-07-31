from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        return requirements
    
setup(
    name = "BackOrderPrediction",
    version = "0.0.1",
    author = "Somto Ogbe",
    author_email = "ogbesomto4@gmail.com",
    # Here we specify packages that we want to be installed before our package is created
    # install_requires = ['pandas','sklearn']
    # Instead of doing these stuffs like this why can't we read the requirement.txt file and install whatever is there
    install_requires=get_requirements('requirements.txt'),
    # This finds all the packages in the project
    packages = find_packages()
)