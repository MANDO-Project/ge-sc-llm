pragma solidity ^0.5.7;
library SafeMath {
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
}
contract ERC20TokenInterface {
    function balanceOf(address _owner) public view returns (uint256 value);
function bug_tmstmp8 () public payable {
	uint pastBlockTime_tmstmp8;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp8);  
        pastBlockTime_tmstmp8 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function transfer(address _to, uint256 _value) public returns (bool success);
address winner_tmstmp39;
function play_tmstmp39(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp39 = msg.sender;}}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
function bug_tmstmp36 () public payable {
	uint pastBlockTime_tmstmp36;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp36);  
        pastBlockTime_tmstmp36 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function approve(address _spender, uint256 _value) public returns (bool success);
address winner_tmstmp35;
function play_tmstmp35(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp35 = msg.sender;}}
    function allowance(address _owner, address _spender) public view returns (uint256 remaining);
function bug_tmstmp40 () public payable {
	uint pastBlockTime_tmstmp40;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp40);  
        pastBlockTime_tmstmp40 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    }
contract ERC20Token is ERC20TokenInterface {
    using SafeMath for uint256;
  function bug_tmstmp32 () public payable {
	uint pastBlockTime_tmstmp32;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp32);  
        pastBlockTime_tmstmp32 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  uint256 public totalSupply;
  address winner_tmstmp38;
function play_tmstmp38(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp38 = msg.sender;}}
  mapping (address => uint256) balances;  
  function bug_tmstmp4 () public payable {
	uint pastBlockTime_tmstmp4;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp4);  
        pastBlockTime_tmstmp4 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  mapping (address => mapping (address => uint256)) allowed;  
    function balanceOf(address _owner) public view returns (uint256 value) {
        return balances[_owner];
    }
function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function transfer(address _to, uint256 _value) public returns (bool success) {
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
address winner_tmstmp27;
function play_tmstmp27(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp27 = msg.sender;}}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(_from, _to, _value);
        return true;
    }
address winner_tmstmp31;
function play_tmstmp31(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp31 = msg.sender;}}
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }
uint256 bugv_tmstmp5 = block.timestamp;
  uint256 bugv_tmstmp3 = block.timestamp;
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  uint256 bugv_tmstmp4 = block.timestamp;
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract AsseteGram is ERC20Token {
  address winner_tmstmp7;
function play_tmstmp7(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp7 = msg.sender;}}
  string public name = 'Electronic Gram';
  address winner_tmstmp23;
function play_tmstmp23(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp23 = msg.sender;}}
  uint8 public decimals = 3;
  address winner_tmstmp14;
function play_tmstmp14(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp14 = msg.sender;}}
  string public symbol = 'eGram';
  address winner_tmstmp30;
function play_tmstmp30(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp30 = msg.sender;}}
  string public version = '2';
    constructor() public {
        address initialOwner = 0xac775cD446889ac167da466692449ece5439fc12;
        totalSupply = 180000000 * (10**uint256(decimals));  
        balances[initialOwner] = totalSupply;
        emit Transfer(address(0), initialOwner, balances[initialOwner]);
    }
uint256 bugv_tmstmp1 = block.timestamp;
    function() external {
        revert();
    }
uint256 bugv_tmstmp2 = block.timestamp;
}