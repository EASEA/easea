/***********************************************************************
| CcsvParser.h                                                          |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2019-05                                                         |
|                                                                       |
 ***********************************************************************/

#pragma once


#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>

// Class of Data
class CcsvData{
    
    std::vector<std::string> data_;
    char separator_;

public:
    CcsvData(const std::string& s, char separator = ',')
    :separator_(separator)
    {
	parse(s);
    }
    
    CcsvData(char separator = ',')
    :separator_(separator){}
    
    void parse(const std::string& s, char separator){
	 separator_ = separator;
	 parse(s);
    }
    
    void parse(const std::string& s){
	size_t i = 0;
	size_t j = 0;
	size_t n = s.length();

	data_.clear();

	for( ; ; ) {
	    for( ; i < n && s[i] != separator_; i++);
	    if(i >= n) {
		data_.push_back(s.substr(j, i - j));
		break;
	    }else {
		data_.push_back(s.substr(j, i - j));
		i ++;
		j = i;
	    }
	}
    }
    
    // returns the number of cells in this data 
    unsigned int size() const {
	return data_.size();
    }
    
    std::vector<std::string> getData(void) const {
	return data_;
    }

    // returns the value of cell of the index, where the index is in the range [0, size() )
    std::string getCell(int index) const {
	return data_[index];
    }

    // the same as getCell, but with the the specification og the data type
    template<typename DataType>
    DataType getCellAs(int index) const {
	std::istringstream input(data_[index]);
	DataType t;
	input >> t;
	
	return t;
    }
    
 
    // data printing    
    void print(std::ostream& out) const {
	for(size_t i = 0, n = data_.size(); i < n; i++){
	    if(i){
		out << separator_;
	    }
	    out << data_[ i ];
	}
    }

};
// overloading of <<
inline std::ostream& operator<<(std::ostream& out, const CcsvData& data){
    data.print(out);
    return out;
}

//Class of Parser
class CcsvParser{
    std::ifstream& input_;
    char separator_;
    CcsvData header_;
    std::vector<CcsvData> data_;
    std::vector<std::string> content_;

public:
    CcsvParser(std::ifstream& input, char separator = ',')
    :input_(input),
    separator_(separator){
	std::string line;
	assert(input_.is_open()!=false);
	    while(input_.good()){
		getline(input_, line);
		if(line != "")
		    content_.push_back(line);
	    }
	
	input_.close();
	assert(content_.size() != 0);
	
	parseHeader();
	parseContent();
    }
    
    void parseHeader(){
	std::stringstream s(content_[0]);
	std::string line;
	while(std::getline(s, line)){
	    if (!line.empty())
		header_.parse(line, separator_);
	}
	    
    }

    void parseContent(){
	std::vector<std::string>::iterator it;
	CcsvData data;
	it = content_.begin();
	it++;

	for(; it != content_.end(); it++){
	    std::string line = *it;
	    if(!line.empty())
	    {
		data.parse(line, separator_);
		data_.push_back(data);
	    }
	}
    }

    const CcsvData& getHeader() const {
	return header_;
    }
    
    CcsvData getRow(unsigned int rowPosition) const {
	assert(rowPosition <= data_.size());
	return data_[rowPosition];
    }
    

    template<typename DataType>
    DataType getCell(unsigned int rowPosition, unsigned int cellPosition) const {
	CcsvData row = getRow(rowPosition);
        DataType t = row.getCellAs<DataType>(cellPosition);
        return t;
    }
    
    unsigned int getColumnNumber(void) const{
	return header_.size();
    }

    unsigned int getRowNumber(void) const {
	return data_.size();
    }

    
    const std::string getHeaderElem(unsigned int pos) const {
	assert(pos < header_.size());
	return header_.getData()[pos];
    }
};

