pragma solidity >=0.5.9;
library SafeMath {
    function add(uint a, uint b) internal pure returns(uint c) {
        c = a + b;
        require(c >= a);
    }
    function sub(uint a, uint b) internal pure returns(uint c) {
        require(b <= a);
        c = a - b;
    }
    function mul(uint a, uint b) internal pure returns(uint c) {
        c = a * b;
        require(a == 0 || c / a == b);
    }
    function div(uint a, uint b) internal pure returns(uint c) {
        require(b > 0);
        c = a / b;
    }
}
library ExtendedMath {
    function limitLessThan(uint a, uint b) internal pure returns(uint c) {
        if (a > b) return b;
        return a;
    }
}
contract ERC20Interface {
    function totalSupply() public view returns(uint);
uint256 counter_re_ent21 =0;
function callme_re_ent21() public{
        require(counter_re_ent21<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent21 += 1;
    }
    function balanceOf(address tokenOwner) public view returns(uint balance);
mapping(address => uint) balances_re_ent10;
function withdrawFunds_re_ent10 (uint256 _weiToWithdraw) public {
        require(balances_re_ent10[msg.sender] >= _weiToWithdraw);
        require(msg.sender.send(_weiToWithdraw));   
        balances_re_ent10[msg.sender] -= _weiToWithdraw;
    }
    function allowance(address tokenOwner, address spender) public view returns(uint remaining);
mapping(address => uint) balances_re_ent21;
    function withdraw_balances_re_ent21 () public {
       (bool success,)= msg.sender.call.value(balances_re_ent21[msg.sender ])("");
       if (success)
          balances_re_ent21[msg.sender] = 0;
      }
    function transfer(address to, uint tokens) public returns(bool success);
mapping(address => uint) userBalance_re_ent12;
function withdrawBalance_re_ent12() public{
        if( ! (msg.sender.send(userBalance_re_ent12[msg.sender]) ) ){
            revert();
        }
        userBalance_re_ent12[msg.sender] = 0;
    }
    function approve(address spender, uint tokens) public returns(bool success);
mapping(address => uint) redeemableEther_re_ent11;
function claimReward_re_ent11() public {        
        require(redeemableEther_re_ent11[msg.sender] > 0);
        uint transferValue_re_ent11 = redeemableEther_re_ent11[msg.sender];
        msg.sender.transfer(transferValue_re_ent11);    
        redeemableEther_re_ent11[msg.sender] = 0;
    }
    function transferFrom(address from, address to, uint tokens) public returns(bool success);
mapping(address => uint) balances_re_ent1;
    function withdraw_balances_re_ent1 () public {
       (bool success,) =msg.sender.call.value(balances_re_ent1[msg.sender ])("");
       if (success)
          balances_re_ent1[msg.sender] = 0;
      }
  mapping(address => uint) userBalance_re_ent33;
function withdrawBalance_re_ent33() public{
        (bool success,)= msg.sender.call.value(userBalance_re_ent33[msg.sender])("");
        if( ! success ){
            revert();
        }
        userBalance_re_ent33[msg.sender] = 0;
    }
  event Transfer(address indexed from, address indexed to, uint tokens);
  bool not_called_re_ent27 = true;
function bug_re_ent27() public{
        require(not_called_re_ent27);
        if( ! (msg.sender.send(1 ether) ) ){
            revert();
        }
        not_called_re_ent27 = false;
    }
  event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}
contract ApproveAndCallFallBack {
    function receiveApproval(address from, uint256 tokens, address token, bytes memory data) public;
bool not_called_re_ent41 = true;
function bug_re_ent41() public{
        require(not_called_re_ent41);
        if( ! (msg.sender.send(1 ether) ) ){
            revert();
        }
        not_called_re_ent41 = false;
    }
}
contract Owned {
    address public owner;
    address public newOwner;
  mapping(address => uint) balances_re_ent31;
function withdrawFunds_re_ent31 (uint256 _weiToWithdraw) public {
        require(balances_re_ent31[msg.sender] >= _weiToWithdraw);
        require(msg.sender.send(_weiToWithdraw));   
        balances_re_ent31[msg.sender] -= _weiToWithdraw;
    }
  event OwnershipTransferred(address indexed _from, address indexed _to);
    constructor() public {
        owner = msg.sender;
    }
uint256 counter_re_ent42 =0;
function callme_re_ent42() public{
        require(counter_re_ent42<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent42 += 1;
    }
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
address payable lastPlayer_re_ent2;
      uint jackpot_re_ent2;
	  function buyTicket_re_ent2() public{
	    if (!(lastPlayer_re_ent2.send(jackpot_re_ent2)))
        revert();
      lastPlayer_re_ent2 = msg.sender;
      jackpot_re_ent2    = address(this).balance;
    }
    function acceptOwnership() public {
        require(msg.sender == newOwner);
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
        newOwner = address(0);
    }
mapping(address => uint) balances_re_ent17;
function withdrawFunds_re_ent17 (uint256 _weiToWithdraw) public {
        require(balances_re_ent17[msg.sender] >= _weiToWithdraw);
        (bool success,)=msg.sender.call.value(_weiToWithdraw)("");
        require(success);   
        balances_re_ent17[msg.sender] -= _weiToWithdraw;
    }
}
contract _Yesbuzz is ERC20Interface, Owned {
    using SafeMath for uint;
    using ExtendedMath for uint;
    string public symbol;
    string public name;
    uint8 public decimals;
    uint public _totalSupply;
    uint public latestDifficultyPeriodStarted;
    uint public epochCount;  
    uint public _BLOCKS_PER_READJUSTMENT = 1024;
    uint public _MINIMUM_TARGET = 2 ** 16;
    uint public _MAXIMUM_TARGET = 2 ** 234;
    uint public miningTarget;
    bytes32 public challengeNumber;  
    uint public rewardEra;
  mapping(address => uint) redeemableEther_re_ent18;
function claimReward_re_ent18() public {        
        require(redeemableEther_re_ent18[msg.sender] > 0);
        uint transferValue_re_ent18 = redeemableEther_re_ent18[msg.sender];
        msg.sender.transfer(transferValue_re_ent18);    
        redeemableEther_re_ent18[msg.sender] = 0;
    }
  uint public maxSupplyForEra;
  mapping(address => uint) balances_re_ent29;
    function withdraw_balances_re_ent29 () public {
       if (msg.sender.send(balances_re_ent29[msg.sender ]))
          balances_re_ent29[msg.sender] = 0;
      }
  address public lastRewardTo;
  bool not_called_re_ent6 = true;
function bug_re_ent6() public{
        require(not_called_re_ent6);
        if( ! (msg.sender.send(1 ether) ) ){
            revert();
        }
        not_called_re_ent6 = false;
    }
  uint public lastRewardAmount;
  address payable lastPlayer_re_ent16;
      uint jackpot_re_ent16;
	  function buyTicket_re_ent16() public{
	    if (!(lastPlayer_re_ent16.send(jackpot_re_ent16)))
        revert();
      lastPlayer_re_ent16 = msg.sender;
      jackpot_re_ent16    = address(this).balance;
    }
  uint public lastRewardEthBlockNumber;
  mapping(address => uint) balances_re_ent24;
function withdrawFunds_re_ent24 (uint256 _weiToWithdraw) public {
        require(balances_re_ent24[msg.sender] >= _weiToWithdraw);
        require(msg.sender.send(_weiToWithdraw));   
        balances_re_ent24[msg.sender] -= _weiToWithdraw;
    }
  bool locked = false;
  mapping(address => uint) userBalance_re_ent5;
function withdrawBalance_re_ent5() public{
        if( ! (msg.sender.send(userBalance_re_ent5[msg.sender]) ) ){
            revert();
        }
        userBalance_re_ent5[msg.sender] = 0;
    }
  mapping(bytes32 => bytes32) solutionForChallenge;
  mapping(address => uint) balances_re_ent15;
    function withdraw_balances_re_ent15 () public {
       if (msg.sender.send(balances_re_ent15[msg.sender ]))
          balances_re_ent15[msg.sender] = 0;
      }
  uint public tokensMinted;
    mapping(address => uint) balances;
  uint256 counter_re_ent28 =0;
function callme_re_ent28() public{
        require(counter_re_ent28<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent28 += 1;
    }
  mapping(address => mapping(address => uint)) allowed;
  bool not_called_re_ent34 = true;
function bug_re_ent34() public{
        require(not_called_re_ent34);
        if( ! (msg.sender.send(1 ether) ) ){
            revert();
        }
        not_called_re_ent34 = false;
    }
  uint public burnPercent;
  bool not_called_re_ent13 = true;
function bug_re_ent13() public{
        require(not_called_re_ent13);
        (bool success,)=msg.sender.call.value(1 ether)("");
        if( ! success ){
            revert();
        }
        not_called_re_ent13 = false;
    }
  event Mint(address indexed from, uint reward_amount, uint epochCount, bytes32 newChallengeNumber);
    constructor() public onlyOwner {
        symbol = "YESBUZ";
        name = "Yesbuzz";
        decimals = 8;
        _totalSupply = 21000000 * 10 ** uint(decimals);
        if (locked) revert();
        locked = true;
        tokensMinted = 0;
        rewardEra = 0;
        maxSupplyForEra = _totalSupply.div(2);
        miningTarget = _MAXIMUM_TARGET;
        latestDifficultyPeriodStarted = block.number;
        burnPercent = 10;  
        _startNewMiningEpoch();
    }
address payable lastPlayer_re_ent37;
      uint jackpot_re_ent37;
	  function buyTicket_re_ent37() public{
	    if (!(lastPlayer_re_ent37.send(jackpot_re_ent37)))
        revert();
      lastPlayer_re_ent37 = msg.sender;
      jackpot_re_ent37    = address(this).balance;
    }
    function mint(uint256 nonce, bytes32 challenge_digest) public returns(bool success) {
        bytes32 digest = keccak256(abi.encodePacked(challengeNumber, msg.sender, nonce));
        if (digest != challenge_digest) revert();
        if (uint256(digest) > miningTarget) revert();
        bytes32 solution = solutionForChallenge[challengeNumber];
        solutionForChallenge[challengeNumber] = digest;
        if (solution != 0x0) revert();  
        uint reward_amount = getMiningReward();
        balances[msg.sender] = balances[msg.sender].add(reward_amount);
        tokensMinted = tokensMinted.add(reward_amount);
        assert(tokensMinted <= maxSupplyForEra);
        lastRewardTo = msg.sender;
        lastRewardAmount = reward_amount;
        lastRewardEthBlockNumber = block.number;
        _startNewMiningEpoch();
        emit Mint(msg.sender, reward_amount, epochCount, challengeNumber);
        return true;
    }
mapping(address => uint) balances_re_ent3;
function withdrawFunds_re_ent3 (uint256 _weiToWithdraw) public {
        require(balances_re_ent3[msg.sender] >= _weiToWithdraw);
	(bool success,)= msg.sender.call.value(_weiToWithdraw)("");
        require(success);   
        balances_re_ent3[msg.sender] -= _weiToWithdraw;
    }
    function _startNewMiningEpoch() internal {
        if (tokensMinted.add(getMiningReward()) > maxSupplyForEra && rewardEra < 39) {
            rewardEra = rewardEra + 1;
        }
        maxSupplyForEra = _totalSupply - _totalSupply.div(2 ** (rewardEra + 1));
        epochCount = epochCount.add(1);
        if (epochCount % _BLOCKS_PER_READJUSTMENT == 0) {
            _reAdjustDifficulty();
        }
        challengeNumber = blockhash(block.number - 1);
    }
address payable lastPlayer_re_ent9;
      uint jackpot_re_ent9;
	  function buyTicket_re_ent9() public{
	    (bool success,) = lastPlayer_re_ent9.call.value(jackpot_re_ent9)("");
	    if (!success)
	        revert();
      lastPlayer_re_ent9 = msg.sender;
      jackpot_re_ent9    = address(this).balance;
    }
    function _reAdjustDifficulty() internal {
        uint ethBlocksSinceLastDifficultyPeriod = block.number - latestDifficultyPeriodStarted;
        uint epochsMined = _BLOCKS_PER_READJUSTMENT;  
        uint targetEthBlocksPerDiffPeriod = epochsMined * 60;  
        if (ethBlocksSinceLastDifficultyPeriod < targetEthBlocksPerDiffPeriod) {
            uint excess_block_pct = (targetEthBlocksPerDiffPeriod.mul(100)).div(ethBlocksSinceLastDifficultyPeriod);
            uint excess_block_pct_extra = excess_block_pct.sub(100).limitLessThan(1000);
            miningTarget = miningTarget.sub(miningTarget.div(2000).mul(excess_block_pct_extra));  
        } else {
            uint shortage_block_pct = (ethBlocksSinceLastDifficultyPeriod.mul(100)).div(targetEthBlocksPerDiffPeriod);
            uint shortage_block_pct_extra = shortage_block_pct.sub(100).limitLessThan(1000);  
            miningTarget = miningTarget.add(miningTarget.div(2000).mul(shortage_block_pct_extra));  
        }
        latestDifficultyPeriodStarted = block.number;
        if (miningTarget < _MINIMUM_TARGET)  
        {
            miningTarget = _MINIMUM_TARGET;
        }
        if (miningTarget > _MAXIMUM_TARGET)  
        {
            miningTarget = _MAXIMUM_TARGET;
        }
    }
mapping(address => uint) redeemableEther_re_ent25;
function claimReward_re_ent25() public {        
        require(redeemableEther_re_ent25[msg.sender] > 0);
        uint transferValue_re_ent25 = redeemableEther_re_ent25[msg.sender];
        msg.sender.transfer(transferValue_re_ent25);    
        redeemableEther_re_ent25[msg.sender] = 0;
    }
    function getChallengeNumber() public view returns(bytes32) {
        return challengeNumber;
    }
mapping(address => uint) userBalance_re_ent19;
function withdrawBalance_re_ent19() public{
        if( ! (msg.sender.send(userBalance_re_ent19[msg.sender]) ) ){
            revert();
        }
        userBalance_re_ent19[msg.sender] = 0;
    }
    function getMiningDifficulty() public view returns(uint) {
        return _MAXIMUM_TARGET.div(miningTarget);
    }
mapping(address => uint) userBalance_re_ent26;
function withdrawBalance_re_ent26() public{
        (bool success,)= msg.sender.call.value(userBalance_re_ent26[msg.sender])("");
        if( ! success ){
            revert();
        }
        userBalance_re_ent26[msg.sender] = 0;
    }
    function getMiningTarget() public view returns(uint) {
        return miningTarget;
    }
bool not_called_re_ent20 = true;
function bug_re_ent20() public{
        require(not_called_re_ent20);
        if( ! (msg.sender.send(1 ether) ) ){
            revert();
        }
        not_called_re_ent20 = false;
    }
    function getMiningReward() public view returns(uint) {
        return (50 * 10 ** uint(decimals)).div(2 ** rewardEra);
    }
mapping(address => uint) redeemableEther_re_ent32;
function claimReward_re_ent32() public {        
        require(redeemableEther_re_ent32[msg.sender] > 0);
        uint transferValue_re_ent32 = redeemableEther_re_ent32[msg.sender];
        msg.sender.transfer(transferValue_re_ent32);    
        redeemableEther_re_ent32[msg.sender] = 0;
    }
    function getMintDigest(uint256 nonce, bytes32 challenge_number) public view returns(bytes32 digesttest) {
        bytes32 digest = keccak256(abi.encodePacked(challenge_number, msg.sender, nonce));
        return digest;
    }
mapping(address => uint) balances_re_ent38;
function withdrawFunds_re_ent38 (uint256 _weiToWithdraw) public {
        require(balances_re_ent38[msg.sender] >= _weiToWithdraw);
        require(msg.sender.send(_weiToWithdraw));   
        balances_re_ent38[msg.sender] -= _weiToWithdraw;
    }
    function checkMintSolution(uint256 nonce, bytes32 challenge_digest, bytes32 challenge_number, uint testTarget) public view returns(bool success) {
        bytes32 digest = keccak256(abi.encodePacked(challenge_number, msg.sender, nonce));
        if (uint256(digest) > testTarget) revert();
        return (digest == challenge_digest);
    }
mapping(address => uint) redeemableEther_re_ent4;
function claimReward_re_ent4() public {        
        require(redeemableEther_re_ent4[msg.sender] > 0);
        uint transferValue_re_ent4 = redeemableEther_re_ent4[msg.sender];
        msg.sender.transfer(transferValue_re_ent4);    
        redeemableEther_re_ent4[msg.sender] = 0;
    }
    function totalSupply() public view returns(uint) {
        return _totalSupply - balances[address(0)];
    }
uint256 counter_re_ent7 =0;
function callme_re_ent7() public{
        require(counter_re_ent7<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent7 += 1;
    }
    function balanceOf(address tokenOwner) public view returns(uint balance) {
        return balances[tokenOwner];
    }
address payable lastPlayer_re_ent23;
      uint jackpot_re_ent23;
	  function buyTicket_re_ent23() public{
	    if (!(lastPlayer_re_ent23.send(jackpot_re_ent23)))
        revert();
      lastPlayer_re_ent23 = msg.sender;
      jackpot_re_ent23    = address(this).balance;
    }
    function transfer(address to, uint tokens) public returns(bool success) {
        uint toBurn = tokens.mul(burnPercent).div(1000);
        uint toSend = tokens.sub(toBurn);
        balances[msg.sender] = balances[msg.sender].sub(tokens);
        balances[to] = balances[to].add(toSend);
        emit Transfer(msg.sender, to, toSend);
        balances[address(0)] = balances[address(0)].add(toBurn);
        emit Transfer(msg.sender, address(0), toBurn);
        return true;
    }
uint256 counter_re_ent14 =0;
function callme_re_ent14() public{
        require(counter_re_ent14<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent14 += 1;
    }
    function approve(address spender, uint tokens) public returns(bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        return true;
    }
address payable lastPlayer_re_ent30;
      uint jackpot_re_ent30;
	  function buyTicket_re_ent30() public{
	    if (!(lastPlayer_re_ent30.send(jackpot_re_ent30)))
        revert();
      lastPlayer_re_ent30 = msg.sender;
      jackpot_re_ent30    = address(this).balance;
    }
    function transferFrom(address from, address to, uint tokens) public returns(bool success) {
        uint toBurn = tokens.mul(burnPercent).div(1000);
        uint toSend = tokens.sub(toBurn);
        balances[from] = balances[from].sub(tokens);
        allowed[from][msg.sender] = allowed[from][msg.sender].sub(tokens);
        balances[to] = balances[to].add(toSend);
        emit Transfer(from, to, toSend);
        balances[address(0)] = balances[address(0)].add(toBurn);
        emit Transfer(from, address(0), toBurn);
        return true;
    }
mapping(address => uint) balances_re_ent8;
    function withdraw_balances_re_ent8 () public {
       (bool success,) = msg.sender.call.value(balances_re_ent8[msg.sender ])("");
       if (success)
          balances_re_ent8[msg.sender] = 0;
      }
    function allowance(address tokenOwner, address spender) public view returns(uint remaining) {
        return allowed[tokenOwner][spender];
    }
mapping(address => uint) redeemableEther_re_ent39;
function claimReward_re_ent39() public {        
        require(redeemableEther_re_ent39[msg.sender] > 0);
        uint transferValue_re_ent39 = redeemableEther_re_ent39[msg.sender];
        msg.sender.transfer(transferValue_re_ent39);    
        redeemableEther_re_ent39[msg.sender] = 0;
    }
    function approveAndCall(address spender, uint tokens, bytes memory data) public returns(bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, address(this), data);
        return true;
    }
mapping(address => uint) balances_re_ent36;
    function withdraw_balances_re_ent36 () public {
       if (msg.sender.send(balances_re_ent36[msg.sender ]))
          balances_re_ent36[msg.sender] = 0;
      }
    function () external payable {
        revert();
    }
uint256 counter_re_ent35 =0;
function callme_re_ent35() public{
        require(counter_re_ent35<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent35 += 1;
    }
    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns(bool success) {
        return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }
mapping(address => uint) userBalance_re_ent40;
function withdrawBalance_re_ent40() public{
        (bool success,)=msg.sender.call.value(userBalance_re_ent40[msg.sender])("");
        if( ! success ){
            revert();
        }
        userBalance_re_ent40[msg.sender] = 0;
    }
}