pragma solidity >=0.5.11; 
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        uint256 c = a / b;
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "SafeMath: modulo by zero");
        return a % b;
    }
}
contract owned {
  address winner_tmstmp22;
function play_tmstmp22(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp22 = msg.sender;}}
  address public owner;
  function bug_tmstmp12 () public payable {
	uint pastBlockTime_tmstmp12;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp12);  
        pastBlockTime_tmstmp12 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  address private newOwner;
  address winner_tmstmp31;
function play_tmstmp31(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp31 = msg.sender;}}
  event OwnershipTransferred(uint256 curTime, address indexed _from, address indexed _to);
    constructor() public {
        owner = msg.sender;
    }
function bug_tmstmp9() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function onlyOwnerTransferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
function bug_tmstmp25() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function acceptOwnership() public {
        require(msg.sender == newOwner);
        emit OwnershipTransferred(now, owner, newOwner);
        owner = newOwner;
        newOwner = address(0);
    }
address winner_tmstmp19;
function play_tmstmp19(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp19 = msg.sender;}}
}
interface ERC20Essential 
{
    function transfer(address _to, uint256 _amount) external returns (bool);
    function transferFrom(address _from, address _to, uint256 _amount) external returns (bool);
}
contract BitpayerDEX is owned {
  using SafeMath for uint256;
address winner_tmstmp11;
function play_tmstmp11(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp11 = msg.sender;}}
  bool public safeGuard;  
function bug_tmstmp1() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  address public feeAccount;  
address winner_tmstmp2;
function play_tmstmp2(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp2 = msg.sender;}}
  uint public tradingFee = 50;  
function bug_tmstmp17() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  mapping (address => mapping (address => uint)) public tokens;  
function bug_tmstmp37() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  mapping (address => mapping (bytes32 => bool)) public orders;  
address winner_tmstmp3;
function play_tmstmp3(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp3 = msg.sender;}}
  mapping (address => mapping (bytes32 => uint)) public orderFills;  
function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  event Order(uint256 curTime, address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user);
uint256 bugv_tmstmp5 = block.timestamp;
  event Cancel(uint256 curTime, address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s);
uint256 bugv_tmstmp1 = block.timestamp;
  event Trade(uint256 curTime, address tokenGet, uint amountGet, address tokenGive, uint amountGive, address get, address give);
uint256 bugv_tmstmp2 = block.timestamp;
  event Deposit(uint256 curTime, address token, address user, uint amount, uint balance);
uint256 bugv_tmstmp3 = block.timestamp;
  event Withdraw(uint256 curTime, address token, address user, uint amount, uint balance);
uint256 bugv_tmstmp4 = block.timestamp;
  event OwnerWithdrawTradingFee(address indexed owner, uint256 amount);
    constructor() public {
        feeAccount = msg.sender;
    }
address winner_tmstmp26;
function play_tmstmp26(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp26 = msg.sender;}}
    function changeSafeguardStatus() onlyOwner public
    {
        if (safeGuard == false)
        {
            safeGuard = true;
        }
        else
        {
            safeGuard = false;    
        }
    }
function bug_tmstmp20 () public payable {
	uint pastBlockTime_tmstmp20;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp20);  
        pastBlockTime_tmstmp20 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function calculatePercentage(uint256 PercentOf, uint256 percentTo ) internal pure returns (uint256) 
    {
        uint256 factor = 10000;
        require(percentTo <= factor);
        uint256 c = PercentOf.mul(percentTo).div(factor);
        return c;
    }
function bug_tmstmp32 () public payable {
	uint pastBlockTime_tmstmp32;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp32);  
        pastBlockTime_tmstmp32 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }  
  function changeFeeAccount(address feeAccount_) public onlyOwner {
    feeAccount = feeAccount_;
  }
address winner_tmstmp38;
function play_tmstmp38(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp38 = msg.sender;}}
  function changetradingFee(uint tradingFee_) public onlyOwner{
    tradingFee = tradingFee_;
  }
function bug_tmstmp4 () public payable {
	uint pastBlockTime_tmstmp4;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp4);  
        pastBlockTime_tmstmp4 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  function availableTradingFeeOwner() public view returns(uint256){
      return tokens[address(0)][feeAccount];
  }
address winner_tmstmp7;
function play_tmstmp7(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp7 = msg.sender;}}
  function withdrawTradingFeeOwner() public onlyOwner returns (string memory){
      uint256 amount = availableTradingFeeOwner();
      require (amount > 0, 'Nothing to withdraw');
      tokens[address(0)][feeAccount] = 0;
      msg.sender.transfer(amount);
      emit OwnerWithdrawTradingFee(owner, amount);
  }
address winner_tmstmp23;
function play_tmstmp23(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp23 = msg.sender;}}
  function deposit() public payable {
    tokens[address(0)][msg.sender] = tokens[address(0)][msg.sender].add(msg.value);
    emit Deposit(now, address(0), msg.sender, msg.value, tokens[address(0)][msg.sender]);
  }
address winner_tmstmp14;
function play_tmstmp14(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp14 = msg.sender;}}
  function withdraw(uint amount) public {
    require(!safeGuard,"System Paused by Admin");
    require(tokens[address(0)][msg.sender] >= amount);
    tokens[address(0)][msg.sender] = tokens[address(0)][msg.sender].sub(amount);
    msg.sender.transfer(amount);
    emit Withdraw(now, address(0), msg.sender, amount, tokens[address(0)][msg.sender]);
  }
address winner_tmstmp30;
function play_tmstmp30(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp30 = msg.sender;}}
  function depositToken(address token, uint amount) public {
    require(token!=address(0));
    require(ERC20Essential(token).transferFrom(msg.sender, address(this), amount));
    tokens[token][msg.sender] = tokens[token][msg.sender].add(amount);
    emit Deposit(now, token, msg.sender, amount, tokens[token][msg.sender]);
  }
function bug_tmstmp8 () public payable {
	uint pastBlockTime_tmstmp8;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp8);  
        pastBlockTime_tmstmp8 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  function withdrawToken(address token, uint amount) public {
    require(!safeGuard,"System Paused by Admin");
    require(token!=address(0));
    require(tokens[token][msg.sender] >= amount);
    tokens[token][msg.sender] = tokens[token][msg.sender].sub(amount);
	  ERC20Essential(token).transfer(msg.sender, amount);
    emit Withdraw(now, token, msg.sender, amount, tokens[token][msg.sender]);
  }
address winner_tmstmp39;
function play_tmstmp39(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp39 = msg.sender;}}
  function balanceOf(address token, address user) public view returns (uint) {
    return tokens[token][user];
  }
function bug_tmstmp36 () public payable {
	uint pastBlockTime_tmstmp36;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp36);  
        pastBlockTime_tmstmp36 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  function order(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce) public {
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    orders[msg.sender][hash] = true;
    emit Order(now, tokenGet, amountGet, tokenGive, amountGive, expires, nonce, msg.sender);
  }
address winner_tmstmp35;
function play_tmstmp35(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp35 = msg.sender;}}
  function trade(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s, uint amount) public {
    require(!safeGuard,"System Paused by Admin");
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    require((
      (orders[user][hash] || ecrecover(keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash)),v,r,s) == user) &&
      block.number <= expires &&
      orderFills[user][hash].add(amount) <= amountGet
    ));
    tradeBalances(tokenGet, amountGet, tokenGive, amountGive, user, amount);
    orderFills[user][hash] = orderFills[user][hash].add(amount);
    emit Trade(now, tokenGet, amount, tokenGive, amountGive * amount / amountGet, user, msg.sender);
  }
  function tradeBalances(address tokenGet, uint amountGet, address tokenGive, uint amountGive, address user, uint amount) internal {
    uint tradingFeeXfer = calculatePercentage(amount,tradingFee);
    tokens[tokenGet][msg.sender] = tokens[tokenGet][msg.sender].sub(amount.add(tradingFeeXfer));
    tokens[tokenGet][user] = tokens[tokenGet][user].add(amount.sub(tradingFeeXfer));
    tokens[address(0)][feeAccount] = tokens[address(0)][feeAccount].add(tradingFeeXfer);
    tokens[tokenGive][user] = tokens[tokenGive][user].sub(amountGive.mul(amount) / amountGet);
    tokens[tokenGive][msg.sender] = tokens[tokenGive][msg.sender].add(amountGive.mul(amount) / amountGet);
  }
function bug_tmstmp40 () public payable {
	uint pastBlockTime_tmstmp40;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp40);  
        pastBlockTime_tmstmp40 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  function testTrade(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s, uint amount, address sender) public view returns(bool) {
    if (!(
      tokens[tokenGet][sender] >= amount &&
      availableVolume(tokenGet, amountGet, tokenGive, amountGive, expires, nonce, user, v, r, s) >= amount
    )) return false;
    return true;
  }
function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  function availableVolume(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s) public view returns(uint) {
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    uint available1;
    if (!(
      (orders[user][hash] || ecrecover(keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash)),v,r,s) == user) &&
      block.number <= expires
    )) return 0;
    available1 = tokens[tokenGive][user].mul(amountGet) / amountGive;
    if (amountGet.sub(orderFills[user][hash])<available1) return amountGet.sub(orderFills[user][hash]);
    return available1;
  }
  function amountFilled(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user) public view returns(uint) {
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    return orderFills[user][hash];
  }
address winner_tmstmp27;
function play_tmstmp27(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp27 = msg.sender;}}
  function cancelOrder(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, uint8 v, bytes32 r, bytes32 s) public {
    require(!safeGuard,"System Paused by Admin");
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    require((orders[msg.sender][hash] || ecrecover(keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash)),v,r,s) == msg.sender));
    orderFills[msg.sender][hash] = amountGet;
    emit Cancel(now, tokenGet, amountGet, tokenGive, amountGive, expires, nonce, msg.sender, v, r, s);
  }
}