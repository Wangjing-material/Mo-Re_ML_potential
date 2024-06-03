 Mo-Re_ML_potential
 ===================
 The full details of the machine learning model, along with the code for LAMMPS implementation are displayed in MLP and tabMLP folders, respectively.
# 1 How to use machine learning potential(MLP)
 1.1 Requires lammps version 3 March 2020 or older!  
 1.2 Copy the files("pair_ml_energy.cpp" and "pair_ml_energy.h") in "MLP" into your lammps/src/ folder and compile normally.  
 1.3 For pure Mo, use the pair_style ml/energy as for example:  
    pair_style      ml/energy 1 -6.10  
    pair_coeff      * * Param_ML_pot.txt Mo  
    for Mo-Re alloy, or:  
    pair_style      ml/energy 1 -6.10  
    pair_coeff      * * Param_ML_pot.txt Mo Re  
 1.4 The file for the MLP model is named "MLP/Param_ML_pot.txt", and you need to place this file in your computation directory. The usability of the potential function can be tested through the "MLP/in.test" file.
 # 2 How to use tabulated machine learning potential(tabMLP)
 1.1 Requires lammps version 24 March 2022 or newer!  
 1.2 Copy the files("pair_tabMLP.cpp" and "pair_tabMLP.h") in "tabMLP" into your lammps/src/ folder and compile normally.  
 1.3 For pure Mo, use the pair_style tabMLP as for example:  
    pair_style      tabMLP  
    pair_coeff      * * Mo-Re.tabMLP Mo yes yes yes  
    for Mo-Re alloy, or:  
    pair_style      tabMLP  
    pair_coeff      * * Mo-Re.tabMLP Mo Re yes yes yes  
 1.4 Since the potential function file (Mo-Re.tabMLP) is too large, you can download it from the following link: https://github.com/Wangjing-material/Mo-Re_ML_potential/blob/master/Mo-Re.tabMLP. Click the "Download raw file" button to download.  
 1.5 After download the "Mo-Re.tabMLP" file, you need to place this file in your computation directory.The usability of the potential function can be tested through the "tabMLP/in.test" file.
