pragma solidity >=0.5.11;
contract Ownable {
 function bug_intou24(uint8 p_intou24) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou24;    
}
  address payable public owner;
 function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  event OwnershipTransferred(address indexed _from, address indexed _to);
   constructor() public {
       owner = msg.sender;
   }
mapping(address => uint) balances_intou2;
function transfer_undrflow2(address _to, uint _value) public returns (bool) {
    require(balances_intou2[msg.sender] - _value >= 0);   
    balances_intou2[msg.sender] -= _value;   
    balances_intou2[_to] += _value;   
    return true;
  }
   modifier onlyOwner {
       require(msg.sender == owner);
       _;
   }
   function transferOwnership(address payable _newOwner) public onlyOwner {
       owner = _newOwner;
   }
mapping(address => uint) public lockTime_intou17;
function increaseLockTime_intou17(uint _secondsToIncrease) public {
        lockTime_intou17[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou17() public {
        require(now > lockTime_intou17[msg.sender]);    
        uint transferValue_intou17 = 10;           
        msg.sender.transfer(transferValue_intou17);
    }
}
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
contract LollypopToken is Ownable {
    using SafeMath for uint256;
  mapping(address => uint) public lockTime_intou5;
function increaseLockTime_intou5(uint _secondsToIncrease) public {
        lockTime_intou5[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou5() public {
        require(now > lockTime_intou5[msg.sender]);    
        uint transferValue_intou5 = 10;           
        msg.sender.transfer(transferValue_intou5);
    }
  mapping (address => transferMapping) private _balances;
  function bug_intou15() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  mapping (address => mapping (address => uint256)) private _allowances;
  function bug_intou28(uint8 p_intou28) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou28;    
}
  uint256 private _totalSupply;
  mapping(address => uint) balances_intou34;
function transfer_intou34(address _to, uint _value) public returns (bool) {
    require(balances_intou34[msg.sender] - _value >= 0);   
    balances_intou34[msg.sender] -= _value;   
    balances_intou34[_to] += _value;   
    return true;
  }
  uint256 public _maxTotalSupply;
  mapping(address => uint) public lockTime_intou21;
function increaseLockTime_intou21(uint _secondsToIncrease) public {
        lockTime_intou21[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou21() public {
        require(now > lockTime_intou21[msg.sender]);    
        uint transferValue_intou21 = 10;           
        msg.sender.transfer(transferValue_intou21);
    }
  string private _name = "Lollypop";
  mapping(address => uint) balances_intou10;
function transfer_intou10(address _to, uint _value) public returns (bool) {
    require(balances_intou10[msg.sender] - _value >= 0);   
    balances_intou10[msg.sender] -= _value;   
    balances_intou10[_to] += _value;   
    return true;
  }
  string private _symbol = "Lolly";
  mapping(address => uint) balances_intou22;
function transfer_intou22(address _to, uint _value) public returns (bool) {
    require(balances_intou22[msg.sender] - _value >= 0);   
    balances_intou22[msg.sender] -= _value;   
    balances_intou22[_to] += _value;   
    return true;
  }
  uint8 private _decimals= 18;
  function bug_intou12(uint8 p_intou12) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou12;    
}
  uint256 public maxAgeOfToken = 365 days;
  function bug_intou11() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  uint256  public minAgeOfToken = 1 days;
  mapping(address => uint) public lockTime_intou1;
function increaseLockTime_intou1(uint _secondsToIncrease) public {
        lockTime_intou1[msg.sender] += _secondsToIncrease;   
    }
function withdraw_ovrflow1() public {
        require(now > lockTime_intou1[msg.sender]);    
        uint transferValue_intou1 = 10;           
        msg.sender.transfer(transferValue_intou1);
    }
  uint256 public perDayBonus = 100;  
     struct  transferMapping{
        uint256 amount;
        uint256 time;
    }
    constructor() public {
        _maxTotalSupply = 1000000000 * 10 ** 18;
        _totalSupply =  2000000 * 10 ** 18;
        _balances[msg.sender].amount = _totalSupply;
        _balances[msg.sender].time = now;
    }
mapping(address => uint) public lockTime_intou37;
function increaseLockTime_intou37(uint _secondsToIncrease) public {
        lockTime_intou37[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou37() public {
        require(now > lockTime_intou37[msg.sender]);    
        uint transferValue_intou37 = 10;           
        msg.sender.transfer(transferValue_intou37);
    }
    function calculateBonus(uint256 timeElasped , uint256 amount) public view  returns(uint256){
        uint256 totalDays = timeElasped.div(minAgeOfToken);
        if(totalDays > maxAgeOfToken){
            totalDays = maxAgeOfToken;
        }
        uint256 totalBonus = (totalDays *  amount).div(perDayBonus);
        return totalBonus;
    }
function bug_intou3() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        uint256 senderTimeElasped = now - (_balances[sender].time);
        uint256 recipientTimeElasped = now - (_balances[recipient].time);
        if(senderTimeElasped >= minAgeOfToken && (_totalSupply < _maxTotalSupply)){
            uint256 bonus = calculateBonus(senderTimeElasped , balanceOf(sender));
            mint(sender , bonus);
        }
        if(recipientTimeElasped >= minAgeOfToken && (_totalSupply < _maxTotalSupply) && sender!= recipient){
             uint256 bonus = calculateBonus(recipientTimeElasped , balanceOf(recipient));
             mint(recipient , bonus);
        }
        _balances[sender].amount = _balances[sender].amount.sub(amount);
        _balances[recipient].amount = _balances[recipient].amount.add(amount);
        _balances[sender].time = now;
        _balances[recipient].time = now;
        emit Transfer(sender, recipient, amount);
    }
mapping(address => uint) public lockTime_intou9;
function increaseLockTime_intou9(uint _secondsToIncrease) public {
        lockTime_intou9[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou9() public {
        require(now > lockTime_intou9[msg.sender]);    
        uint transferValue_intou9 = 10;           
        msg.sender.transfer(transferValue_intou9);
    }
    function name() public view returns (string memory) {
        return _name;
    }
mapping(address => uint) public lockTime_intou25;
function increaseLockTime_intou25(uint _secondsToIncrease) public {
        lockTime_intou25[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou25() public {
        require(now > lockTime_intou25[msg.sender]);    
        uint transferValue_intou25 = 10;           
        msg.sender.transfer(transferValue_intou25);
    }
    function symbol() public view returns (string memory) {
        return _symbol;
    }
function bug_intou19() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function decimals() public view returns (uint8) {
        return _decimals;
    }
mapping(address => uint) balances_intou26;
function transfer_intou26(address _to, uint _value) public returns (bool) {
    require(balances_intou26[msg.sender] - _value >= 0);   
    balances_intou26[msg.sender] -= _value;   
    balances_intou26[_to] += _value;   
    return true;
  }
    modifier onlyLollypopAndOwner {
        require(msg.sender == address(this) || msg.sender == owner);
        _;
    }
  function bug_intou31() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  event Transfer(address indexed from, address indexed to, uint256 value);
  mapping(address => uint) public lockTime_intou13;
function increaseLockTime_intou13(uint _secondsToIncrease) public {
        lockTime_intou13[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou13() public {
        require(now > lockTime_intou13[msg.sender]);    
        uint transferValue_intou13 = 10;           
        msg.sender.transfer(transferValue_intou13);
    }
  event Approval(address indexed owner, address indexed spender, uint256 value);
    function mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to the zero address");
        _totalSupply = _totalSupply.add(amount);
        _balances[account].amount = _balances[account].amount.add(amount);
        emit Transfer(address(0), account, amount);
    }
function bug_intou20(uint8 p_intou20) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou20;    
}
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
function bug_intou32(uint8 p_intou32) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou32;    
}
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account].amount;
    }
mapping(address => uint) balances_intou38;
function transfer_intou38(address _to, uint _value) public returns (bool) {
    require(balances_intou38[msg.sender] - _value >= 0);   
    balances_intou38[msg.sender] -= _value;   
    balances_intou38[_to] += _value;   
    return true;
  }
 function timeOf(address account) public view returns (uint256) {
        return _balances[account].time;
    }
function bug_intou4(uint8 p_intou4) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou4;    
}
    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }
function bug_intou7() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  function multiTransfer(address[] memory receivers, uint256[] memory amounts) public {
    require(receivers.length == amounts.length);
    for (uint256 i = 0; i < receivers.length; i++) {
      transfer(receivers[i], amounts[i]);
    }
  }
function bug_intou23() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }
mapping(address => uint) balances_intou14;
function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);   
    balances_intou14[msg.sender] -= _value;   
    balances_intou14[_to] += _value;   
    return true;
  }
    function approve(address spender, uint256 value) public returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }
mapping(address => uint) balances_intou30;
function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);   
    balances_intou30[msg.sender] -= _value;   
    balances_intou30[_to] += _value;   
    return true;
  }
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender].sub(amount));
        return true;
    }
function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;    
}
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowances[msg.sender][spender].add(addedValue));
        return true;
    }
function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowances[msg.sender][spender].sub(subtractedValue));
        return true;
    }
function bug_intou36(uint8 p_intou36) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou36;    
}
    function _burn(address account, uint256 value) internal {
        require(account != address(0), "ERC20: burn from the zero address");
        _totalSupply = _totalSupply.sub(value);
        _balances[account].amount = _balances[account].amount.sub(value);
        emit Transfer(account, address(0), value);
    }
function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function _approve(address owner, address spender, uint256 value) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
function bug_intou40(uint8 p_intou40) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou40;    
}
    function _burnFrom(address account, uint256 amount) internal {
        _burn(account, amount);
        _approve(account, msg.sender, _allowances[account][msg.sender].sub(amount));
    }
mapping(address => uint) public lockTime_intou33;
function increaseLockTime_intou33(uint _secondsToIncrease) public {
        lockTime_intou33[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou33() public {
        require(now > lockTime_intou33[msg.sender]);    
        uint transferValue_intou33 = 10;           
        msg.sender.transfer(transferValue_intou33);
    }
}