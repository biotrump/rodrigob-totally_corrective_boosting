
# include "parse.hpp"


namespace totally_corrective_boosting
{

void throw_if_fail(std::istream &in, std::string expected) throw()
{
    if (in.fail())
    {
        throw std::string("Error before trying to look for " + expected);
    }
}

void throw_if_eof(std::istream &in, const std::string &expected) throw()
{
    if (in.eof())
    {
        throw std::string("Error before trying to look for " + expected);
    }
}

void expect_keyword(std::istream& in, const std::string &kw) throw(std::string)
{
    std::string s;
    throw_if_eof(in, "token '" + kw + "'");
    throw_if_fail(in, "token '" + kw + "'");
    in >> s;
    throw_if_fail(in, "token '" + kw + "'");
    if (s != kw)
    {
        throw std::string("Expected keyword '" + kw + "' but got '" + s + "'");
    }
}

int expect_int(std::istream& in) throw(std::string)
{
    int i;
    throw_if_eof(in, "int");
    throw_if_fail(in, "int");
    in >> i;
    throw_if_fail(in, "int");
    return i;
}

std::string expect_word(std::istream& in) throw(std::string)
{
    std::string s;
    throw_if_eof(in, "string");
    throw_if_fail(in, "string");
    in >> s;
    throw_if_fail(in, "string");
    return s;
}

void chomp_input_until(std::istream& in, const std::string &kw) throw(std::string)
{
    std::string s;
    while (s != kw)
    {
        if (in.eof())
        {
            throw std::string("Unexpected EOF waiting for '" + kw + "'");
        }
        in >> s;
    }
}

} // end of namespace totally_corrective_boosting

