from setuptools import find_packages,setup

HYPHEN_E_DOT = '-e .'

def get_requirements():
    """
    This function will return list of requirements
    """

    requirement_list= []

    with open('requirements.txt') as f:
        lines = f.readlines()
        for requirement in lines:
            requirement = str.split(requirement,sep='\n')[0] 
            if requirement == HYPHEN_E_DOT:
                continue
            requirement_list.append(requirement)
    
    return requirement_list

setup(
    name = "sensor",
    version = "0.0.1",
    author = "eliyas",
    author_email = "eliyaschozhakkad@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements(),#['pymongo==4.2.0'],
)
