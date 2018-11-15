# from data_qf file folder

# use 2006-2008 as test data
ls level2/201*UTC/d*/*/*npy > list_of_data.txt

ls level2/200*UTC/d*/*/*npy > list_of_validate_data.txt

python refine_data_filename_list.py
