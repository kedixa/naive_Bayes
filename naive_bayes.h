/*
 * Copyright (C) Kedixa Liu
 *  kedixa@outlook.com
 * naive Bayes 算法学习
 *
 */

#include<iostream>
#include<algorithm>
#include<vector>
#include<string>
#include<unordered_map>
#include<cassert>

#ifndef NAIVE_BAYES_H_
#define NAIVE_BAYES_H_

namespace kedixa{

class naive_bayes;
class _naive_bayes_node
{
	bool                is_num;     // 是否为数值型数据
	double              mean_value; // 均值
	double              variance;   // 方差
	std::vector<double> p_attr;     // 非数值型数据保存概率
public:
	_naive_bayes_node()
	{
		is_num     = false;
		mean_value = 0.0;
		variance   = 0.0;
	}
	friend class naive_bayes;
};

class naive_bayes
{
	typedef std::vector<std::string>            vs;
	typedef std::vector<vs>                     vvs;
	typedef std::vector<int>                    vi;
	typedef std::vector<vi>                     vvi;
	typedef std::unordered_map<std::string,int> usi;
	typedef std::vector<usi>                    vusi;
	typedef _naive_bayes_node                   node;
	typedef std::vector<node>                   vn;
	typedef std::vector<vn>                     vvn;
	typedef std::vector<bool>                   vb;
	typedef std::vector<double>                 vd;
private:
	double      m_estimate;      // m-估计
	std::string target_attr;     // 目标属性
	vs          headers;         // 各个属性的名称
	vusi        attr_to_int;     // 属性到整数的映射
	vvs         int_to_attr;     // 整数到属性的映射
	vi          attrs_size;      // 每个属性有多少不同的属性值
	vb          is_numeric;      // 属性是否为数值型
	vvs         datas;           // 保存原始数据
	vvn         p_datas;         // 保存各概率值
	vi          target_to_label; // 目标属性值的下标
	vd          p_target;        // 每个目标属性值的概率
	int         num_attr;        // 属性数量
	int         num_data;        // 数据集大小
	int         num_targ;        // 目标属性有多少不同的取值


public:
	naive_bayes();
	bool set_data(vvs&, vs&, vb b = vb());
	bool clear(); // 使用clear释放空间
	bool run();
	std::string classification(vs&);
	~naive_bayes();
};

} // namespace kedixa
#endif // NAIVE_BAYES_H_
