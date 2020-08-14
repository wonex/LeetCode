#include "Solution.h"

bool _isMatch(string, string, int, int, map<string, bool>&);

bool Solution::isMatch(string s, string p)
{
	map<string, bool> memo;
	return _isMatch(s, p, 0, 0 ,memo);
}
bool _isMatch(string s, string p, int i, int j, map<string, bool>& memo)
{
	string key = to_string(i) + to_string(j);
	if (memo.find(key) != memo.end()) { return memo[key]; }
	bool ans;
	if (j == p.size()) { return i == s.size(); }
	bool firstMatch = i < s.size() && (p[j] == s[i] || p[j] == '.');
	if (p.size() - j >= 2 && p[j + 1] == '*')
	{
		ans = _isMatch(s, p, i, j + 2, memo) || (firstMatch && _isMatch(s, p, i + 1, j, memo));
	}
	else
		ans = firstMatch && _isMatch(s, p, i + 1, j + 1, memo);
	memo[key] = ans;
	return ans;
}

bool Solution::isMatch(string s, string p, bool is_dp = true)
{
	if (s.size() == 0) { return p.size() == 0; }
	vector<vector<bool>> dp(s.size(), vector<bool>(p.size(),false));
	dp[0][0] = true;
	for (int i = 1; i <= s.size(); ++i)
	{
		for (int j = 1; j <= p.size(); ++j)
		{
			if (s[i-1] == p[j-1] || '.' == p[j-1])
				dp[i][j] = dp[i-1][j-1];
			else if ('*' == p[j-1])
			{
				if (s[i-1] == p[j - 2])
				{
					dp[i][j] = dp[i - 1][j - 1];
					--j;
				}
				else
				{
					dp[i][j] = dp[i - 1][j - 1];
					--i;
				}
			}
		}
	}
}

void traceback(vector<int>& nums, int i, int sum, int S, int& ans)
{

	if (i == nums.size())
	{
		if (sum == S)
			++ans;
		return;
	}
	traceback(nums, i + 1, sum - nums[i], S, ans);
	traceback(nums, i + 1, sum + nums[i], S, ans);
}
int Solution::findTargetSumWays(vector<int>& nums, int S) {
	int ans = 0;
	traceback(nums, 0, 0, S, ans);
	return ans;
}

int traceback(vector<int>& nums, int i, int sum, int S, map<string, int> memo)
{
	if (i == nums.size()) { return sum == S ? 1 : 0; }
	string key;
	key = to_string(i) + to_string(sum);
	if (memo.find(key) != memo.end())
		return memo[key];
	int ans = traceback(nums, i + 1, sum - nums[i], S, memo) + \
		traceback(nums, i + 1, sum + nums[i], S, memo);
	memo[key] = ans;
	return ans;
}
int Solution::findTargetSumWays(vector<int>& nums, int S, bool useMemo=true) {
	map<string, int> memo;
	int ans = traceback(nums, 0, 0, S, memo);
	return ans;
}

int Solution::findTargetSumWays(vector<int>& nums, int S, int is_dp=1) {
	int sum = accumulate(nums.begin(), nums.end(), 0);
	if (sum < S || (sum + S) % 2 == 1) {
		return 0;
	}
	int target = (sum + S) / 2;
	int m = nums.size();
	vector<vector<int>> dp(m + 1, vector<int>(target + 1));
	dp[0][0] = 1;
	for (int i = 1; i <= m; ++i)
	{
		for (int j = 0; j <= target; ++j)
		{
			if (j - nums[i - 1] >= 0)
			{
				dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]];
			}
			else
				dp[i][j] = dp[i - 1][j];
		}
	}
	return dp[m][target];
}

ListNode* Solution::reverseList(ListNode* head) {
	if (head == NULL || head->next == NULL)
		return head;
	ListNode* node = reverseList(head->next);//返回最右边的Node
	head->next->next = head;//翻转后的链表中最左边的Node
	head->next = NULL;
	return node;
}

ListNode* Solution::reverseList(ListNode* head, bool twoPointer) {
	if (head == NULL || head->next == NULL)
		return head;
	ListNode* pre = NULL;
	ListNode* cur = head;
	ListNode* tmp = NULL;
	while (cur != NULL)//必须是cur，而不能是cur->next，不然最后一个Node没有处理
	{
		tmp = cur->next;
		cur->next = pre;
		pre = cur;
		cur = tmp;
	}
	return pre;//cur指向NULL，因此返回pre
}

//最长回文子串，关键在于对当前的判断依赖于前面的结果
//只有里面的是回文子串的时候，判断当前是否是回文串才有意义
string Solution::longestPalindrome(string s, bool use_dp) {
	if (s.size() <= 1)
		return s;
	int n = s.size();
	vector<vector<bool>> dp(n, vector<bool>(n));
	int begin, end;
	begin = 0;
	end = 0;
	for (int i = n - 1; i >= 0; --i)
	{
		for (int j = i; j < n; ++j)
		{
			if (j - i <= 1)
				dp[i][j] = s[i] == s[j];
			else
				dp[i][j] = dp[i + 1][j - 1] && s[i] == s[j];
			if (dp[i][j] && j - i > end - begin)
			{
				begin = i;
				end = j;
			}
		}
	}
	return s.substr(begin, end - begin + 1);
}
//利用里面子串的回文子串长度来判断其是否为回文子串。
//关键在于初始化为0，以及对于长度为1和2的子串的处理;
//其实可以利用子串长度与其回文子串长度是否匹配来判断是否为回文子串。
string Solution::longestPalindrome(string s, int use_dp) {
	int m = s.size();
	vector<vector<int>> dp(m + 1, vector<int>(m + 1, 0));
	int res = 1;
	int begin = 0;
	for (int i = m - 2; i >= 0; --i)
	{
		dp[i][i] = 1;
		if (s[i] == s[i + 1])
		{
			dp[i][i + 1] = 2;
			if (res < 2)
			{
				res = 2;
				begin = i;
			}
		}
		for (int j = i + 2; j < m; ++j)
		{
			if (s[i] == s[j] && dp[i + 1][j - 1] >= 1)
			{
				dp[i][j] = dp[i + 1][j - 1] + 2;
				if (dp[i][j] > res)
				{
					res = dp[i][j];
					begin = i;
				}
			}
		}
	}
	return s.substr(begin, res);
}
//对于每一个点，向两边扩展。
//分为奇数扩展和偶数扩展
string Solution::longestPalindrome(string s) {
	if (s.size() <= 1)
		return s;
	int n = s.size();
	int begin, end;
	begin = 0;
	end = 0;
	for (int i = 0; i < n; ++i)
	{
		int j = 1;
		while (i - j >= 0 and i + j < n)
		{
			if (s[i - j] == s[i + j])
				++j;
			else
				break;
		}
		if (2 * j - 1 > end - begin)
		{
			begin = i - j + 1;
			end = i + j - 1;
		}

		j = 0;
		while (i - j >= 0 and i + 1 + j < n)
		{
			if (s[i - j] == s[i + j + 1])
				++j;
			else
				break;
		}
		if (2 * j > end - begin)
		{
			begin = i - j + 1;
			end = i + j;
		}
	}
	return s.substr(begin, end - begin + 1);
}

//最长回文子序列，对当前的判断不依赖于里面的结果
//因为回文子序列不要求连续
int longestPalindromeSubseq(string s) {
	int m = s.size();
	vector<vector<int>> dp(m + 1, vector<int>(m + 1, 0));
	for (int i = m - 1; i >= 0; --i)
	{
		dp[i][i] = 1;
		for (int j = i + 1; j < m; ++j)
		{
			if (s[i] == s[j])
				dp[i][j] = dp[i + 1][j - 1] + 2;
			else
				dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
		}
	}
	return dp[0][m - 1];
}

//一种O(n^2)复杂度的算法
//对于当前的点，利用栈来判断以此为起点的最长匹配括号有多少
int Solution::longestValidParentheses(string s) {
	if (s.size() <= 1) { return 0; }
	int begin = -1;
	int length = 0;
	for (int i = 0; i < s.size(); ++i)
	{
		stack<char> sta;
		int j = i;
		int res = 0;
		while (j < s.size())
		{
			if (s[j] == '(')
				sta.push(s[j]);
			else
			{
				if (sta.empty())
					break;
				sta.pop();
			}
			++j;
			if (sta.empty())
				res = j - i;
		}
		if (res > length)
		{
			length = res;
			begin = i;
		}
	}
	return length;
}
//对当前括号的匹配情况，依赖于前一个括号，以及前一个括号的匹配情况
int Solution::longestValidParentheses(string s, bool use_dp) {
	if (s.size() <= 1)
		return 0;
	int n = s.size();
	vector<int> dp(n + 1, 0);
	int res = 0;
	for (int i = 2; i <= n; ++i)
	{
		if (s[i - 1] == ')')
		{
			if (s[i - 2] == '(')
				dp[i] = dp[i - 2] + 2;
			else
			{
				if (i - dp[i - 1] - 2 >= 0 && s[i - dp[i - 1] - 2] == '(')
					dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2];
			}
			res = max(dp[i], res);
		}
	}
	return res;
}

//无限硬币的关键点在于，选择取当前物品的状态转移来自于上方而不是左上方
//即只是减少了总量，而没有减少硬币的选择。
int Solution::coinChange(vector<int>& coins, int amount) {
	int m = coins.size();
	int n = amount;
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, amount + 1));//初始化小技巧
	for (int i = 0; i <= m; ++i)
		dp[i][0] = 0;
	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (coins[i - 1] <= j)
				dp[i][j] = min(dp[i - 1][j], dp[i][j - coins[i - 1]] + 1);
			else
				dp[i][j] = dp[i - 1][j];
		}
	}
	return dp[m][n] > amount ? -1 : dp[m][n];
}
int Solution::coinChange(int amount, vector<int>& coins) {
	int m = coins.size();
	vector<int> dp(amount + 1, amount + 1);
	dp[0] = 0;
	for (int i = 0; i < m; ++i)
		for (int j = coins[i]; j <= amount; ++j)
		{
			dp[j] = min(dp[j - coins[i]] + 1, dp[j]);
		}
	return dp[amount] > amount ? -1 : dp[amount];
}

int Solution::change(int amount, vector<int>& coins) {
	int m = amount;
	int n = coins.size();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	for (int i = 0; i <= n; ++i)
		dp[0][i] = 1;
	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (coins[j - 1] <= i)
				dp[i][j] = dp[i][j - 1] + dp[i - coins[j - 1]][j];
			else
				dp[i][j] = dp[i][j - 1];
		}
	}
	return dp[m][n];
}
int Solution::change(vector<int>& coins, int amount) {
	int m = coins.size();
	vector<int> dp(amount + 1, 0);
	dp[0] = 1;
	for (int i = 0; i < m; ++i)
		for (int j = coins[i]; j <= amount; ++j)
			dp[j] += dp[j - coins[i]];
	return dp[amount];

}

//核心在于开区间和最后一个被戳破的气球，其实就是动态规划状态的定义
//技巧性的地方在于在数组首尾各增加一个1
int Solution::maxCoins(vector<int>& nums) {
	nums.insert(nums.begin(), 1);
	nums.push_back(1);
	int n = nums.size();
	vector<vector<int>> dp(n + 1, vector<int>(n + 1));
	for (int i = n - 1; i >= 0; --i)
	{
		for (int j = i + 2; j < n; ++j)
		{
			for (int k = i + 1; k < j; ++k)
			{
				dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[j] * nums[k]);
			}
		}
	}
	return dp[0][n - 1];
}

//核心是动态规划状态的定义
//状态定义：假设有j个鸡蛋，且最多可以扔m次的时候，最多能测多少层楼？
//状态转移：扔了一个鸡蛋，那么剩余可扔次数为m-1；如果鸡蛋碎了，则可以测楼下的，为dp[j-1][m-1];
//如果鸡蛋没碎，则可以测楼上的，为dp[j][m-1]；那么dp[j][m]可测的总楼层数为两者相加，再加上本层；
//注意，不是两者选一个，因为鸡蛋碎，或者没碎，也就是当前扔鸡蛋的结果，就可以使楼上或楼下为测过的了。
//所以，其实是测过的和没测过的相加，楼下和楼上，一为测过的，一为没测过的。
int Solution::superEggDrop(int K, int N) {
	vector<vector<int>> dp(K + 1, vector<int>(N + 1, 0));
	if (K == 1) { return N; }
	int m = 0;
	while (dp[K][m] < N)
	{
		++m;
		for (int j = 1; j <= K; ++j)
		{
			dp[j][m] = dp[j][m - 1] + dp[j - 1][m - 1] + 1;
		}
	}
	return m;
}
//递归版动态规划
int dp(int K, int N, vector<vector<int>> memo)
{
	if (memo[K][N] != -1)
	{
		return memo[K][N];
	}
	if (K == 1)
		return N;
	if (N == 0)
		return 0;
	int res = N;
	for (int i = 1; i <= N; ++i)
	{
		res = min(res, max(dp(K - 1, i - 1, memo) + 1, dp(K, N - i, memo) + 1));
	}
	memo[K][N] = res;
	return res;
}
int Solution::superEggDrop(int K, int N, bool use_dp) {
	vector<vector<int>> memo(K + 1, vector<int>(N + 1, -1));
	return dp(K, N, memo);
}
//从下而上的动态规划
int Solution::superEggDrop(int K, int N, int use_dp) {
	vector<vector<int>> dp(K + 1, vector<int>(N + 1, 0));
	for (int i = 1; i <= K; ++i)
	{
		for (int j = 1; j <= N; ++j)
		{
			if (i == 1)
				dp[i][j] = j;
			else
			{
				dp[i][j] = j;
				for (int k = 1; k <= j; ++k)
					dp[i][j] = min(dp[i][j], max(dp[i - 1][k - 1] + 1, dp[i][j - k] + 1));
			}
		}
	}
	return dp[K][N];
}