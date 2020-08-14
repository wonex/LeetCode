#pragma once
#ifndef _SOLUTION_H
#define _SOLUTION_H

#include <string>
#include <map>
#include <vector>
#include <numeric>
#include <stack>
#include <algorithm>

using std::string;
using std::map;
using std::to_string;
using std::vector;
using std::accumulate;
using std::stack;
using std::max;
using std::min;


 //Definition for singly-linked list.
 struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
 };
 

class Solution
{
public:
	bool isMatch(string, string);//10
	bool isMatch(string, string, bool);

	int findTargetSumWays(vector<int>&, int);//494
	int findTargetSumWays(vector<int>&, int, bool);
	int findTargetSumWays(vector<int>&, int, int);

	ListNode* reverseList(ListNode*);//206
	ListNode* reverseList(ListNode*, bool);

	string longestPalindrome(string, bool);//5
	string longestPalindrome(string, int);
	string longestPalindrome(string);

	int longestValidParentheses(string);//32
	int longestValidParentheses(string, bool);

	int coinChange(vector<int>&, int);//322
	int coinChange(int, vector<int>&);

	int change(int, vector<int>&);//518
	int change(vector<int>& coins, int amount);

	int maxCoins(vector<int>&);//312

	int superEggDrop(int, int);//887
	int superEggDrop(int, int, bool);
	int superEggDrop(int, int, int);
};

#endif