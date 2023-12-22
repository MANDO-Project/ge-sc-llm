pragma solidity ^0.5.11;
library SafeMath {
    function mul(uint a, uint b) internal pure returns (uint c) {
        c = a * b;
        require(a == 0 || c / a == b);
    }
   function div(uint a, uint b) internal pure returns (uint c) {
        require(b > 0);
        c = a / b;
    }
    function sub(uint a, uint b) internal pure returns (uint c) {
        require(b <= a);
        c = a - b;
    }
    function add(uint a, uint b) internal pure returns (uint c) {
        c = a + b;
        require(c >= a);
    }
}
contract ERC20Interface {
    function totalSupply() public view returns (uint256);
function bug_tmstmp12 () public payable {
	uint pastBlockTime_tmstmp12;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp12);  
        pastBlockTime_tmstmp12 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function balanceOf(address tokenOwner) public view returns (uint256 balance);
address winner_tmstmp11;
function play_tmstmp11(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp11 = msg.sender;}}
    function allowance(address tokenOwner, address spender) public view returns (uint256 remaining);
function bug_tmstmp1() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function transfer(address to, uint256 tokens) public returns (bool success);
address winner_tmstmp2;
function play_tmstmp2(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp2 = msg.sender;}}
    function approve(address spender, uint256 tokens) public returns (bool success);
function bug_tmstmp17() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function transferFrom(address from, address to, uint256 tokens) public returns (bool success);
function bug_tmstmp37() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  event Transfer(address indexed from, address indexed to, uint256 tokens);
  uint256 bugv_tmstmp5 = block.timestamp;
  event Approval(address indexed tokenOwner, address indexed spender, uint256 tokens);
}
contract Owned {
  function bug_tmstmp16 () public payable {
	uint pastBlockTime_tmstmp16;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp16);  
        pastBlockTime_tmstmp16 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  address payable public owner;
  uint256 bugv_tmstmp1 = block.timestamp;
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor() public {
        owner = msg.sender;
    }
address winner_tmstmp3;
function play_tmstmp3(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp3 = msg.sender;}}
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address payable newOwner) public onlyOwner {
        require(newOwner != address(0));
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
function bug_tmstmp9() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
}
contract ExclusivePlatform is ERC20Interface, Owned {
    using SafeMath for uint256;
  function bug_tmstmp24 () public payable {
	uint pastBlockTime_tmstmp24;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp24);  
        pastBlockTime_tmstmp24 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  mapping (address => uint256) balances;
  function bug_tmstmp5() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  mapping (address => mapping (address => uint256)) allowed;
  address winner_tmstmp15;
function play_tmstmp15(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp15 = msg.sender;}}
  string public name = "Exclusive Platform";
  function bug_tmstmp28 () public payable {
	uint pastBlockTime_tmstmp28;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp28);  
        pastBlockTime_tmstmp28 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  string public symbol = "XPL";
  address winner_tmstmp34;
function play_tmstmp34(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp34 = msg.sender;}}
  uint256 public decimals = 8;
  function bug_tmstmp21() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  uint256 public _totalSupply;
  address winner_tmstmp10;
function play_tmstmp10(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp10 = msg.sender;}}
  uint256 public XPLPerEther = 8000000e8;
    uint256 public minimumBuy = 1 ether / 100;
  address winner_tmstmp22;
function play_tmstmp22(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp22 = msg.sender;}}
  bool public crowdsaleIsOn = true;
    modifier onlyPayloadSize(uint size) {
        assert(msg.data.length >= size + 4);
        _;
    }
    constructor () public {
        _totalSupply = 10000000000e8;
        balances[owner] = _totalSupply;
        emit Transfer(address(0), owner, _totalSupply);
    }
function bug_tmstmp25() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
address winner_tmstmp19;
function play_tmstmp19(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp19 = msg.sender;}}
    function updateXPLPerEther(uint _XPLPerEther) public onlyOwner {        
        emit NewPrice(owner, XPLPerEther, _XPLPerEther);
        XPLPerEther = _XPLPerEther;
    }
address winner_tmstmp26;
function play_tmstmp26(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp26 = msg.sender;}}
    function switchCrowdsale() public onlyOwner {
        crowdsaleIsOn = !(crowdsaleIsOn);
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
    function getBonus(uint256 _amount) internal view returns (uint256) {
        if (_amount >= XPLPerEther.mul(5)) {
            return ((20 * _amount).div(100)).add(_amount);  
        } else if (_amount >= XPLPerEther) {
            return ((5 * _amount).div(100)).add(_amount);  
        }
        return _amount;
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
    function () payable external {
        require(crowdsaleIsOn && msg.value >= minimumBuy);
        uint256 totalBuy =  (XPLPerEther.mul(msg.value)).div(1 ether);
        totalBuy = getBonus(totalBuy);
        doTransfer(owner, msg.sender, totalBuy);
    }
address winner_tmstmp38;
function play_tmstmp38(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp38 = msg.sender;}}
    function distribute(address[] calldata _addresses, uint256 _amount) external {        
        for (uint i = 0; i < _addresses.length; i++) {transfer(_addresses[i], _amount);}
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
    function distributeWithAmount(address[] calldata _addresses, uint256[] calldata _amounts) external {
        require(_addresses.length == _amounts.length);
        for (uint i = 0; i < _addresses.length; i++) {transfer(_addresses[i], _amounts[i]);}
    }
address winner_tmstmp7;
function play_tmstmp7(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp7 = msg.sender;}}
    function doTransfer(address _from, address _to, uint _amount) internal {
        require((_to != address(0)));
        require(_amount <= balances[_from]);
        balances[_from] = balances[_from].sub(_amount);
        balances[_to] = balances[_to].add(_amount);
        emit Transfer(_from, _to, _amount);
    }
address winner_tmstmp23;
function play_tmstmp23(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp23 = msg.sender;}}
    function balanceOf(address _owner) view public returns (uint256) {
        return balances[_owner];
    }
address winner_tmstmp14;
function play_tmstmp14(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp14 = msg.sender;}}
    function transfer(address _to, uint256 _amount) onlyPayloadSize(2 * 32) public returns (bool success) {
        doTransfer(msg.sender, _to, _amount);
        return true;
    }
address winner_tmstmp30;
function play_tmstmp30(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp30 = msg.sender;}}
    function transferFrom(address _from, address _to, uint256 _amount) onlyPayloadSize(3 * 32) public returns (bool success) {
        require(allowed[_from][msg.sender] >= _amount);
        allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_amount);
        doTransfer(_from, _to, _amount);
        return true;
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
    function approve(address _spender, uint256 _amount) public returns (bool success) {
        require((_amount == 0) || (allowed[msg.sender][_spender] == 0));
        allowed[msg.sender][_spender] = _amount;
        emit Approval(msg.sender, _spender, _amount);
        return true;
    }
address winner_tmstmp39;
function play_tmstmp39(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp39 = msg.sender;}}
    function allowance(address _owner, address _spender) view public returns (uint256) {
        return allowed[_owner][_spender];
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
    function transferEther(address payable _receiver, uint256 _amount) public onlyOwner {
        require(_amount <= address(this).balance);
        emit TransferEther(address(this), _receiver, _amount);
        _receiver.transfer(_amount);
    }
address winner_tmstmp35;
function play_tmstmp35(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp35 = msg.sender;}}
    function withdrawFund() onlyOwner public {
        uint256 balance = address(this).balance;
        owner.transfer(balance);
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
    function burn(uint256 _value) onlyOwner public {
        require(_value <= balances[msg.sender]);
        address burner = msg.sender;
        balances[burner] = balances[burner].sub(_value);
        _totalSupply = _totalSupply.sub(_value);
        emit Burn(burner, _value);
    }
function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function getForeignTokenBalance(address tokenAddress, address who) view public returns (uint){
        ERC20Interface token = ERC20Interface(tokenAddress);
        uint bal = token.balanceOf(who);
        return bal;
    }
address winner_tmstmp27;
function play_tmstmp27(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp27 = msg.sender;}}
    function withdrawForeignTokens(address tokenAddress) onlyOwner public returns (bool) {
        ERC20Interface token = ERC20Interface(tokenAddress);
        uint256 amount = token.balanceOf(address(this));
        return token.transfer(owner, amount);
    }
address winner_tmstmp31;
function play_tmstmp31(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp31 = msg.sender;}}
  uint256 bugv_tmstmp2 = block.timestamp;
  event TransferEther(address indexed _from, address indexed _to, uint256 _value);
  uint256 bugv_tmstmp3 = block.timestamp;
  event NewPrice(address indexed _changer, uint256 _lastPrice, uint256 _newPrice);
  uint256 bugv_tmstmp4 = block.timestamp;
  event Burn(address indexed _burner, uint256 value);
}