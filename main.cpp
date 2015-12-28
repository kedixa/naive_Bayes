#include<iostream>
#include<algorithm>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<cmath>
#include "naive_bayes.h"
using namespace std;

/*
 * function: read_data
 * data_file_name: 保存数据的文件名
 * header: 数据的属性名
 * examples: 每行保存一个样例
 *
 */
bool read_data(
		const string& data_file_name,
		vector<string>& header,
		vector<vector<string> >& examples)
{
	// 打开文件
	ifstream input(data_file_name.c_str());

	// 读入属性名
	string line, tmp;
	getline(input, line);
	istringstream iss(line);
	while(iss>>tmp)
		header.push_back(tmp);
	// 保证有数据
	if(header.empty())
	{
		input.close();
		return false;
	}

	// 读入数据集
	while(getline(input, line))
	{
		istringstream iss(line);
		vector<string> v;
		while(iss>>tmp)
			v.push_back(tmp);
		examples.push_back(v);
		// 保证数据属性个数正确
		if(examples.back().size() != header.size())
		{
			input.close();
			return false;
		}
	}

	input.close();
	return true;
}
int main()
{
	vector<string> header;
	vector<vector<string> > datas;

	string data_file_name = "data.txt";
	read_data(data_file_name, header, datas);

	naive_bayes nb;
	nb.set_data(datas, header);
	nb.run();
	vector<string> v={
		"Sunny",
		"Cool",
		"High",
		"Strong"
	};
	cout<<nb.classification(v);
	return 0;
}
