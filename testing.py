#------------------------------------------------------------------------------#
# Authors: Max Chan and Sophie Chen                                            #
#------------------------------------------------------------------------------#

import os

# Save the current working directory
original_cwd = os.getcwd()

# Change the working directory to the "GCNN vs CNN" folder
os.chdir("GCNN vs CNN")

# Execute the testing.py script
os.system("python testing.py")

# Restore the original working directory
os.chdir(original_cwd)

# Change the working directory to the "GCNN vs CNN" folder
os.chdir("GCNN vs CNN with Data Augmentation")

# Execute the testing.py script
os.system("python testing.py")
