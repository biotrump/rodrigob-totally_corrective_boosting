
#ifndef _PARSE_HPP_
#define _PARSE_HPP_

#include <iostream>
#include <string.h>
#include <fstream>
#include <stdlib.h>


namespace totally_corrective_boosting
{

/// Helper functions for parsing the output files

void throw_if_fail(std::istream &in, std::string expected) throw();

void throw_if_eof(std::istream &in, const std::string &expected) throw();
void expect_keyword(std::istream& in, const std::string &kw) throw(std::string);

int expect_int(std::istream& in) throw(std::string);

std::string expect_word(std::istream& in) throw(std::string);

void chomp_input_until(std::istream& in, const std::string &kw) throw(std::string);

} // end of namespace totally_corrective_boosting

#endif
