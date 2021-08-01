import os
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

this_dir = os.path.dirname(os.path.abspath(__file__))
pip_requirements = parse_requirements(
    os.path.join(this_dir, "requirements.txt"), PipSession())

pip_requirements_matscholar_2020v1 = parse_requirements(
    os.path.join(this_dir, "requirements-matscholar_2020v1.txt"), PipSession())

pip_requirements_matbert_ner_2021v1 = parse_requirements(
    os.path.join(this_dir, "requirements-matbert_ner_2021v1.txt"), PipSession()
)

reqs = [pii.requirement for pii in pip_requirements]

reqs_matscholar_2020v1 = [pii.requirement for pii in pip_requirements_matscholar_2020v1]

reqs_matbert_ner_2021v1 = [pii.requirement for pii in pip_requirements_matbert_ner_2021v1]


extras_dict = {
    "matscholar_2020v1": reqs_matscholar_2020v1,
    "matbert_ner_2021v1": reqs_matbert_ner_2021v1
}

readme_path = os.path.join(this_dir, "README.md")

with open(readme_path, "r") as f:
    long_description = f.read()

setup(
    name='lbnlp',
    version='0.1',
    description='Common text mining tools for materials data',
    long_description=long_description,
    url='https://github.com/materialsintelligence/lbnlp',
    author='Alex Dunn',
    author_email='ardunn@lbl.gov',
    license='modified BSD',
    packages=find_packages(),
    package_data={
        "lbnlp.models": ["*.json"]
    },
    zip_safe=False,
    install_requires=reqs,
    extras_require=extras_dict,
    classifiers=['Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.6',
                 'Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: System Administrators',
                 'Intended Audience :: Information Technology',
                 'Operating System :: OS Independent',
                 'Topic :: Other/Nonlisted Topic',
                 'Topic :: Scientific/Engineering'],
    test_suite='lbnlp',
    # tests_require=extras_list,
    scripts=[]
)
