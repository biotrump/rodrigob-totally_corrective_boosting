#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert the files IDA Benchmark files in the current folder,
into the libsvm equivalents.

http://www.raetschlab.org/Members/raetsch/benchmark
"""

from __future__ import print_function

from glob import glob


def find_matching_file_name(pattern):
     files_found = glob(pattern)

     if not files_found:
         raise Exception("Found zero files in the current folder " \
                         "that match the IDA pattern %s" % pattern)
     
     assert len(files_found) == 1, \
             "Found more than one file that match the pattern %s, " \
             "not good. %s" % (pattern, files_found)
     return files_found[0]     

       
def merge_data_and_labels_into_libsvm_file(data_file, labels_file, libsvm_file):
    
    num_lines = 0
    while True:
        label_line = labels_file.readline()
        data_line = data_file.readline()
        
        if not label_line and not data_line:
            break
        elif label_line is None or data_line is None:
            raise Exception("Label file and data file do not have the same lenght")
            
        label = "%+i" % int(float(label_line)) # for the sign, for nicer looking files
        data = [ "%i:%s" % e for e in enumerate(data_line.split(), 1)]
        
        libsvm_line = " ".join([label] + data) + "\n"
        
        libsvm_file.write(libsvm_line)
        
        num_lines += 1
    # end of "while true"
    
    return num_lines


def main():
    
    splits_indices = range(1, 100+1)
    
    data_labels_patterns = []
    for i in splits_indices:
        data_labels_pattern = \
            ("*_train_data_%i.asc" %i, "*_train_labels_%i.asc" %i, "_train_%i.libsvm.txt" %i)
        data_labels_patterns.append( data_labels_pattern )    

    # two for loop make the final output more readable 
    # (all train files, then test files)        
    for i in splits_indices:        
        data_labels_pattern = \
            ("*_test_data_%i.asc" %i, "*_test_labels_%i.asc" %i, "_test_%i.libsvm.txt" %i)
        data_labels_patterns.append( data_labels_pattern )    
        
    for data_labels_pattern in data_labels_patterns:
        
        data_pattern, labels_pattern, libsvm_pattern = data_labels_pattern
        data_file_name = find_matching_file_name(data_pattern)        
        labels_file_name = find_matching_file_name(labels_pattern)        
                
        data_base_name = data_file_name[:-len(data_pattern) + 1]
        labels_base_name = labels_file_name[:-len(labels_pattern) + 1]
        assert data_base_name == labels_base_name, \
                "Basename of data versus labels do not match, not good." \
                "%s != %s" % (data_base_name, labels_base_name)
                
        base_name = data_base_name
        libsvm_file_name = base_name + libsvm_pattern
        
        data_file = open(data_file_name, "r")
        labels_file = open(labels_file_name, "r")
        libsvm_file = open(libsvm_file_name, "w")
        
        num_lines = merge_data_and_labels_into_libsvm_file(data_file, labels_file, libsvm_file)
        
        libsvm_file.close()
        print("File %s created, contains %i samples." % (libsvm_file_name, num_lines))
    # end of "for each pattern"
    
    return

if __name__ == '__main__':
    main()

