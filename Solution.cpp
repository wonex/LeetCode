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
	ListNode* node = reverseList(head->next);//�������ұߵ�Node
	head->next->next = head;//��ת�������������ߵ�Node
	head->next = NULL;
	return node;
}

ListNode* Solution::reverseList(ListNode* head, bool twoPointer) {
	if (head == NULL || head->next == NULL)
		return head;
	ListNode* pre = NULL;
	ListNode* cur = head;
	ListNode* tmp = NULL;
	while (cur != NULL)//������cur����������cur->next����Ȼ���һ��Nodeû�д���
	{
		tmp = cur->next;
		cur->next = pre;
		pre = cur;
		cur = tmp;
	}
	return pre;//curָ��NULL����˷���pre
}

//������Ӵ����ؼ����ڶԵ�ǰ���ж�������ǰ��Ľ��
//ֻ��������ǻ����Ӵ���ʱ���жϵ�ǰ�Ƿ��ǻ��Ĵ���������
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
//���������Ӵ��Ļ����Ӵ��������ж����Ƿ�Ϊ�����Ӵ���
//�ؼ����ڳ�ʼ��Ϊ0���Լ����ڳ���Ϊ1��2���Ӵ��Ĵ���;
//��ʵ���������Ӵ�������������Ӵ������Ƿ�ƥ�����ж��Ƿ�Ϊ�����Ӵ���
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
//����ÿһ���㣬��������չ��
//��Ϊ������չ��ż����չ
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

//����������У��Ե�ǰ���жϲ�����������Ľ��
//��Ϊ���������в�Ҫ������
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

//һ��O(n^2)���Ӷȵ��㷨
//���ڵ�ǰ�ĵ㣬����ջ���ж��Դ�Ϊ�����ƥ�������ж���
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
//�Ե�ǰ���ŵ�ƥ�������������ǰһ�����ţ��Լ�ǰһ�����ŵ�ƥ�����
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

//����Ӳ�ҵĹؼ������ڣ�ѡ��ȡ��ǰ��Ʒ��״̬ת���������Ϸ����������Ϸ�
//��ֻ�Ǽ�������������û�м���Ӳ�ҵ�ѡ��
int Solution::coinChange(vector<int>& coins, int amount) {
	int m = coins.size();
	int n = amount;
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, amount + 1));//��ʼ��С����
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

//�������ڿ���������һ�������Ƶ�������ʵ���Ƕ�̬�滮״̬�Ķ���
//�����Եĵط�������������β������һ��1
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

//�����Ƕ�̬�滮״̬�Ķ���
//״̬���壺������j������������������m�ε�ʱ������ܲ���ٲ�¥��
//״̬ת�ƣ�����һ����������ôʣ����Ӵ���Ϊm-1������������ˣ�����Բ�¥�µģ�Ϊdp[j-1][m-1];
//�������û�飬����Բ�¥�ϵģ�Ϊdp[j][m-1]����ôdp[j][m]�ɲ����¥����Ϊ������ӣ��ټ��ϱ��㣻
//ע�⣬��������ѡһ������Ϊ�����飬����û�飬Ҳ���ǵ�ǰ�Ӽ����Ľ�����Ϳ���ʹ¥�ϻ�¥��Ϊ������ˡ�
//���ԣ���ʵ�ǲ���ĺ�û�������ӣ�¥�º�¥�ϣ�һΪ����ģ�һΪû����ġ�
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
//�ݹ�涯̬�滮
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
//���¶��ϵĶ�̬�滮
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