
/* Copyright (c) 2009, S V N Vishwanathan
 * All rights reserved. 
 * 
 * The contents of this file are subject to the Mozilla Public License 
 * Version 1.1 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at 
 * http://www.mozilla.org/MPL/ 
 * 
 * Software distributed under the License is distributed on an "AS IS" 
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the 
 * License for the specific language governing rights and limitations 
 * under the License. 
 * 
 * Authors: Karen Glocer
 *
 * Created: (09/05/2009) 
 *
 * Last Updated: (09/05/2008)   
 */

#ifndef _PARSE_HPP_
#define _PARSE_HPP_

#include <iostream>
#include <string.h>
#include <fstream>
#include <stdlib.h>

/** Helper functions for parsing the output files */ 

void throw_if_fail(std::istream &in, std::string expected) throw();

void throw_if_eof(std::istream &in, const std::string &expected) throw();
void expect_keyword(std::istream& in, const std::string &kw) throw(std::string);

int expect_int(std::istream& in) throw(std::string);

std::string expect_word(std::istream& in) throw(std::string);

void chomp_input_until(std::istream& in, const std::string &kw) throw(std::string);

#endif
