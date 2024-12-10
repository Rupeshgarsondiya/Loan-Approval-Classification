'''
Author  :Rupesh Garsondiya
github : @Rupeshgarsondiya
Organization : L.J University
'''




from setuptools import find_packages
from setuptools import setup


setup(
  name = "Bank-Loan-Approval-Project",
  version = "1.0",
  description= "to classifiy loan approval based on the customer's credit score",
  author = "Rupesh Garsondiya",
  author_email = "rupeshgarsondiya.edu@gmail.com",
  github = "https://github.com/Rupeshgarsondiya/Loan-Approval-Classification",
  packages = find_packages(),
  install_requires = ['pandas','numpy','sklearn','tensorflow','matplotlib','os'],
  classifiers=[
      "programming Language :: python :: 3"],
   python_requires=">3.6",  
    entry_points={  
        'console_scripts': [  
            'Bank-Loan-Approval-Project-cli=Bank-Loan-Approval-Project.cli:main',  
        ],  
    },  
    test_suite='tests',
)
