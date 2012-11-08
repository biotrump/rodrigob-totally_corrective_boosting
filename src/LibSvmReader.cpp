

#include "LibSvmReader.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdexcept>

namespace totally_corrective_boosting
{

bool LibSVMReader::is_blank(const std::string& line)
{

    for(size_t i = 0; i < line.size(); i++)
        if(!std::isspace(line[i]))
            return false;
    return true;
}

int LibSVMReader::readlibSVM(const std::string& filename,
                             std::vector<SparseVector>& data,
                             std::vector<int>& labels)
{

    std::ifstream df;
    df.open(filename.c_str());
    if(!df.good())
    {
        std::stringstream os;
        os <<"Cannot open data file : " << filename << std::endl;
        throw std::invalid_argument(os.str());
    }

    std::string line;
    size_t max_index = 0;        // maximum index
    size_t tot_nnz = 0;        // number of non zero elements

    // Iterate over lines of a file
    while(!df.eof())
    {

        getline(df, line);

        if(is_blank(line)) break;

        std::istringstream iss(line);

        // Read and store label
        int pt_label = 0;
        iss >> pt_label;
        labels.push_back(pt_label);

        std::string token;
        std::vector<double> pt_val;
        std::vector<size_t> pt_index;

        while(!iss.eof())
        {
            token.clear();
            iss >> token;

            if(is_blank(token)) break;

            size_t index;
            double val;
            sscanf(token.c_str(), "%lu:%lf", &index, &val);

            pt_val.push_back(val);
            pt_index.push_back(index);

            if(index > max_index) max_index = index;
            tot_nnz ++;
        }

        // Create a new data point and push it back into the data vector
        SparseVector data_pt;
        data_pt.nnz = pt_val.size();
        data_pt.val = new double[data_pt.nnz];
        data_pt.index = new size_t[data_pt.nnz];
        size_t i =0;
        for (dbl_itr it = pt_val.begin(); it!=pt_val.end(); it++, i++)
            data_pt.val[i] = *it;
        i = 0;
        for (uint_itr it = pt_index.begin(); it!=pt_index.end(); it++, i++)
            data_pt.index[i] = *it;
        data.push_back(data_pt);
    }

    max_index++;

    std::cout << "Data File: " << filename << std::endl;
    std::cout << "# of data points " << data.size() << std::endl;
    std::cout << "Maximum index " << max_index << std::endl;
    std::cout << "Non zero elements " << tot_nnz << std::endl;
    std::cout << "Total elements " << data.size()*max_index << std::endl;
    std::cout << "Sparsity " << 1.0 - tot_nnz/(1.0*data.size()*max_index) << std::endl;

    // adjust the dimensions
    for (svec_itr it = data.begin(); it!=data.end(); it++)
        (*it).dim = max_index;

    df.close();

    return 0;
}



int LibSVMReader::readlibSVM_transpose(const std::string& filename,
                                       std::vector<SparseVector>& data,
                                       std::vector<int>& labels)
{

    std::ifstream df;
    df.open(filename.c_str());
    if(!df.good())
    {
        std::stringstream os;
        os <<"Cannot open data file : " << filename << std::endl;
        throw std::invalid_argument(os.str());
    }

    std::string line;
    size_t max_index = 0;        // maximum index
    size_t tot_nnz = 0;        // number of non zero elements

    // Iterate over lines of a file
    // Find maximum index
    // Total number of data points
    // Total number of non-zero elements
    // Read labels
    while(!df.eof())
    {

        getline(df, line);

        if(is_blank(line)) break;

        std::istringstream iss(line);

        // Read and store label
        int pt_label = 0;
        iss >> pt_label;
        labels.push_back(pt_label);

        std::string token;

        while(!iss.eof())
        {
            token.clear();
            iss >> token;

            if(is_blank(token)) break;

            size_t index;
            double val;
            sscanf(token.c_str(), "%lu:%lf", &index, &val);

            if(index > max_index) max_index = index;
            tot_nnz ++;
        }
    }

    size_t num_pt = labels.size();

    max_index++;

    std::vector<size_t> index_counter(max_index);

    df.clear();
    df.seekg(0, std::ios::beg);

    // Iterate again over lines of the file
    // This time to find out number of data points per feature
    while(!df.eof())
    {

        getline(df, line);
        if(is_blank(line)) break;

        std::istringstream iss(line);

        // Read label
        int pt_label = 0;
        iss >> pt_label;

        std::string token;

        while(!iss.eof())
        {
            token.clear();
            iss >> token;

            if(is_blank(token)) break;

            size_t index;
            double val;
            sscanf(token.c_str(), "%lu:%lf", &index, &val);
            index_counter[index]++;
        }
    }


    data.reserve(max_index);

    for(size_t i = 0; i < max_index; i++)
    {
        data.push_back(SparseVector(num_pt, index_counter[i]));
        index_counter[i] = 0;
    }

    df.clear();
    df.seekg(0, std::ios::beg);

    size_t pt_num = 0;

    // Finally iterate to load data
    while(!df.eof())
    {

        getline(df, line);

        if(is_blank(line)) break;

        std::istringstream iss(line);

        // Read label
        int pt_label = 0;
        iss >> pt_label;

        std::string token;

        while(!iss.eof())
        {
            token.clear();
            iss >> token;

            if(is_blank(token)) break;

            size_t index;
            double val;
            sscanf(token.c_str(), "%lu:%lf", &index, &val);

            // We need index_counter to keep track of how many spots we have
            // already used in data[index] sparse vector thus far

            // The index is nothing but the data point # that we observed
            data[index].index[index_counter[index]] = pt_num;

            // Value is what we read in
            data[index].val[index_counter[index]]= val;

            index_counter[index]++;
        }

        pt_num++;
    }

    std::cout << "Data File: " << filename << std::endl;
    std::cout << "# of data points " << labels.size() << std::endl;
    std::cout << "Maximum index " << max_index << std::endl;
    std::cout << "Non zero elements " << tot_nnz << std::endl;
    std::cout << "Total elements " << data.size()*pt_num << std::endl;
    std::cout << "Sparsity " << 1.0 - tot_nnz/(1.0*data.size()*pt_num) << std::endl;

    df.close();

    return 0;
}


int LibSVMReader::readlibSVM_transpose_fast(const std::string& filename,
        std::vector<SparseVector>& data,
        std::vector<int>& labels)
{



    std::ifstream df;
    df.open(filename.c_str());
    if(!df.good())
    {
        std::cout<<"Cannot open data file : " << filename << std::endl;
        return 1;
    }

    std::string line;
    size_t max_col = 0;        // maximum hypothesis index
    size_t tot_nnz = 0;        // number of non zero elements


    // vectors of point value and its associated row and col
    size_t max_row = 0;
    std::vector<double> pt;
    std::vector<size_t> row;
    std::vector<size_t> col;


    // Iterate over lines of a file
    while(!df.eof())
    {

        getline(df, line);

        if(is_blank(line)) break;

        std::istringstream iss(line);

        // Read and store label
        int pt_label = 0;
        iss >> pt_label;
        labels.push_back(pt_label);

        std::string token;

        while(!iss.eof())
        {
            token.clear();
            iss >> token;

            if(is_blank(token)) break;

            size_t index;
            double val;

            sscanf(token.c_str(), "%lu:%lf", &index, &val);
            pt.push_back(val);
            col.push_back(index);
            row.push_back(max_row);

            if(index > max_col) max_col = index;

            tot_nnz ++;
        }

        max_row++;
    }
    max_col++;


    // create vector of vectors
    std::vector< std::vector< std::pair<size_t, double> > > tmpmat(max_col);

    for(size_t j = 0; j < col.size(); j++)
    {

        std::pair<size_t,double> tmppair(row[j],pt[j]);
        size_t col_index = col[j];
        tmpmat[col_index].push_back(tmppair);
    }

    // test vector of vectors
    std::vector< std::vector< std::pair<size_t, double> > >::iterator it;
    std::vector< std::pair<size_t, double> >::iterator it2;

    // iterate over the hypotheses
    size_t i = 0;
    for(it = tmpmat.begin(); it != tmpmat.end(); it++,i++)
    {
        // create a hypothesis
        SparseVector hyp;
        hyp.nnz = it->size();
        hyp.val = new double[hyp.nnz];
        hyp.index = new size_t[hyp.nnz];
        hyp.dim = max_row;

        size_t j=0;
        for(it2 = it->begin(); it2 != it->end(); it2++,j++)
        {
            hyp.index[j] = it2->first;
            hyp.val[j] = it2->second;
        }

        data.push_back(hyp);

    }


    std::cout << "Data File: " << filename << std::endl;
    std::cout << "# of hypotheses " << data.size() << std::endl;
    std::cout << "# of examples " << max_row << std::endl;
    std::cout << "Non zero elements " << tot_nnz << std::endl;
    std::cout << "Total elements " << data.size()*max_row << std::endl;
    std::cout << "Sparsity " << 1.0 - tot_nnz/(1.0*data.size()*max_row) << std::endl;

    df.close();

    return 0;
}

} // end of namespace totally_corrective_boosting
