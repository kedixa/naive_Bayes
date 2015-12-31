/*
 * Copyright (C) Kedixa Liu
 *  kedixa@outlook.com
 * naive Bayes 算法学习
 *
 */

#include "naive_bayes.h"
#include<cmath>

namespace kedixa{

naive_bayes::naive_bayes()
{
	m_estimate = 1;
}

bool naive_bayes::clear()
{
	headers.clear();
	attr_to_int.clear();
	int_to_attr.clear();
	attrs_size.clear();
	is_numeric.clear();
	datas.clear();
	p_datas.clear();
	target_to_label.clear();
	p_target.clear();
	return true;
}

/*
 * function: naive_bayes::set_data 设置数据集，产生辅助数据
 * d: 数据集
 * h: 属性, 需保证最后一列为目标属性，且为离散值
 * b: 属性是离散值(false), 还是数值型(true)
 *
 */
bool naive_bayes::set_data(vvs& d, vs& h, vb b)
{
	bool f = clear();
	if(!f) return f;
	
	assert(d.size() > 0); // 数据集不能为空
	datas      = d;
	headers    = h;
	num_attr   = (int)headers.size();
	num_data   = (int)d.size();
	is_numeric = b;
	is_numeric.resize(num_attr, false);
	assert(is_numeric.back() == false); // 目标属性必须为离散值
	target_attr = headers.back();

	attr_to_int.resize(num_attr);
	int_to_attr.resize(num_attr);
	attrs_size.resize(num_attr);

	for(int i = 0; i < num_data; ++i)
	{
		auto& e = d[i];
		for(int j = 0; j < num_attr; ++j)
		{
			if(is_numeric[j]) continue; // 数值型数据不需要映射
			auto it = attr_to_int[j].find(e[j]);
			if(it == attr_to_int[j].end())
			{
				attr_to_int[j][e[j]] = (int)int_to_attr[j].size();
				int_to_attr[j].push_back(e[j]);
			}
		}
	}
	for(int i = 0; i < num_attr; ++i)
		attrs_size[i] = (int)int_to_attr[i].size();
	num_targ = attrs_size.back();

	// 目标属性值的下标
	for(int i = 0; i < num_data; ++i)
		target_to_label.push_back(attr_to_int[num_attr - 1][d[i][num_attr - 1]]);
	return true;
}

/*
 * function: naive_bayes::run  获取各部分概率值，为分类做好准备
 *
 */
bool naive_bayes::run()
{
	p_target.resize(num_targ);
	p_datas.resize(num_targ); // 每行表示一个目标属性值
	for(int i = 0; i < num_targ; ++i)
		p_datas[i].resize(num_attr - 1); // 每列表示一个非目标属性

	for(int k = 0; k < num_targ; ++k)
	{// 对每个目标属性值
		vi data_k; // 目标属性值下标为k的数据集
		for(int i = 0; i < num_data; ++i)
			if(target_to_label[i] == k)
				data_k.push_back(i);

		p_target[k] = (double)data_k.size() / num_data; // 计算每个属性值的概率
		for(int j = 0; j < num_attr - 1; ++j)
		{// 对每个非目标属性
			int k_size = (int)data_k.size();
			auto& p = p_datas[k][j]; // 当前需要计算的结点
			if(is_numeric[j])
			{
				// 计算均值和方差
				double mean_value = 0, variance = 0, sum = 0;
				vd tmp;
				tmp.resize(k_size);
				for(int i = 0; i < k_size; ++i)
					tmp[i] = std::stod(datas[data_k[i]][j]), sum += tmp[i];
				mean_value = sum / k_size;
				for(int i = 0; i < k_size; ++i)
					variance += (tmp[i] - mean_value) * (tmp[i] - mean_value);
				variance /= k_size;
				// 保存结果
				p.is_num = true;
				p.mean_value = mean_value;
				p.variance = variance;
				continue;
			}

			// else 
			p.is_num = false;
			p.p_attr.clear();
			p.p_attr.resize(attrs_size[j], 0.0);
			// 计算每个属性值的数量,即有多少条数据具有该属性值
			for(int i = 0; i < k_size; ++i)
			{
				int tmp = attr_to_int[j][datas[data_k[i]][j]];
				p.p_attr[tmp]++;
			}
			// 计算概率
			double pp = 1.0 / attrs_size[j]; // 用于m-估计的先验概率
			for(int i = 0; i < attrs_size[j]; ++i)
				p.p_attr[i] = (p.p_attr[i] + m_estimate * pp) / (k_size + m_estimate);
			//end if
		}
	}
	return true;
}

/*
 * function: naive_bayes::classification 对数据进行分类
 * return: 该数据最可能的目标属性值
 *
 */
std::string naive_bayes::classification(vs& data)
{
	assert((int)data.size() == num_attr - 1);
	// 为了防止溢出，以下对概率值取了对数
	int max_index = -1;
	double p_max = -1e300; // 最大概率
	vd p_targ_val; // 每个目标属性值对该数据的概率 P(data | target_attr[i])
	p_targ_val.resize(num_targ, 0.0);
	auto f = [&](double x, double u, double d)
	{// 求正态分布概率密度
		return std::exp(-(x - u) * (x - u) / (2 * d)) / sqrt(4 * std::acos(-1) * d);
	};
	for(int i = 0; i < num_targ; ++i)
	{
		auto& t = p_targ_val[i];
		t = std::log(p_target[i]); // 取对数
		for(int j = 0; j < num_attr - 1; ++j)
		{
			auto& p = p_datas[i][j];
			if(is_numeric[j])
			{
				t += std::log(f(std::stod(data[j]), p.mean_value, p.variance));
			}
			else
			{
				auto it = attr_to_int[j].find(data[j]);
				if(it == attr_to_int[j].end())
				{
					std::cerr<<"No such attribute value."<<std::endl;
					exit(1);
				}
				t += std::log(p.p_attr[it->second]);
			}
		}
	}
	// 找到最大概率值
	for(int i = 0; i < num_targ; ++i)
	{
//		std::cout<<p_targ_val[i]<<std::endl;
		if(p_max < p_targ_val[i])
			p_max = p_targ_val[i], max_index = i;
	}
	return int_to_attr[num_attr - 1][max_index];
}

naive_bayes::~naive_bayes()
{
	clear();
}

} // namespace kedixa
