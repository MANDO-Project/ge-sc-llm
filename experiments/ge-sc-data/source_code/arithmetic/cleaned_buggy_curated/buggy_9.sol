pragma solidity >=0.4.22 <0.6.0;
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    if (a == 0) {
        return 0;
    }
    c = a * b;
    assert(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    return a / b;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}
contract Ownable {
  mapping(address => uint) public lockTime_intou17;
function increaseLockTime_intou17(uint _secondsToIncrease) public {
        lockTime_intou17[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou17() public {
        require(now > lockTime_intou17[msg.sender]);    
        uint transferValue_intou17 = 10;           
        msg.sender.transfer(transferValue_intou17);
    }
  address public owner;
    constructor() public {
        owner = msg.sender;
    }
function bug_intou32(uint8 p_intou32) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou32;    
}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
}
contract TokenERC20 is Ownable {
    using SafeMath for uint256;
  mapping(address => uint) public lockTime_intou37;
function increaseLockTime_intou37(uint _secondsToIncrease) public {
        lockTime_intou37[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou37() public {
        require(now > lockTime_intou37[msg.sender]);    
        uint transferValue_intou37 = 10;           
        msg.sender.transfer(transferValue_intou37);
    }
  string public name;
  function bug_intou3() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  string public symbol;
  mapping(address => uint) public lockTime_intou9;
function increaseLockTime_intou9(uint _secondsToIncrease) public {
        lockTime_intou9[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou9() public {
        require(now > lockTime_intou9[msg.sender]);    
        uint transferValue_intou9 = 10;           
        msg.sender.transfer(transferValue_intou9);
    }
  uint8 public decimals;
  mapping(address => uint) public lockTime_intou25;
function increaseLockTime_intou25(uint _secondsToIncrease) public {
        lockTime_intou25[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou25() public {
        require(now > lockTime_intou25[msg.sender]);    
        uint transferValue_intou25 = 10;           
        msg.sender.transfer(transferValue_intou25);
    }
  uint256 private _totalSupply;
  function bug_intou19() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  uint256 public cap;
  mapping(address => uint) balances_intou26;
function transfer_intou26(address _to, uint _value) public returns (bool) {
    require(balances_intou26[msg.sender] - _value >= 0);   
    balances_intou26[msg.sender] -= _value;   
    balances_intou26[_to] += _value;   
    return true;
  }
  mapping (address => uint256) private _balances;
  function bug_intou20(uint8 p_intou20) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou20;    
}
  mapping (address => mapping (address => uint256)) private _allowed;
  function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  event Transfer(address indexed from, address indexed to, uint256 value);
  function bug_intou31() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  event Approval(address indexed owner, address indexed spender, uint256 value);
  mapping(address => uint) public lockTime_intou13;
function increaseLockTime_intou13(uint _secondsToIncrease) public {
        lockTime_intou13[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou13() public {
        require(now > lockTime_intou13[msg.sender]);    
        uint transferValue_intou13 = 10;           
        msg.sender.transfer(transferValue_intou13);
    }
  event Mint(address indexed to, uint256 amount);
    modifier onlyPayloadSize(uint size) {
        require(msg.data.length >= size + 4);
        _;
    }
    constructor(
        uint256 _cap,
        uint256 _initialSupply,
        string memory _name,
        string memory _symbol,
        uint8 _decimals
    ) public {
        require(_cap >= _initialSupply);
        cap = _cap;
        name = _name;                                        
        symbol = _symbol;                                    
        decimals = _decimals;                                
        _totalSupply = _initialSupply;                       
        _balances[owner] = _totalSupply;                     
        emit Transfer(address(0), owner, _totalSupply);
    }
mapping(address => uint) balances_intou38;
function transfer_intou38(address _to, uint _value) public returns (bool) {
    require(balances_intou38[msg.sender] - _value >= 0);   
    balances_intou38[msg.sender] -= _value;   
    balances_intou38[_to] += _value;   
    return true;
  }
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
function bug_intou4(uint8 p_intou4) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou4;    
}
    function balanceOf(address _owner) public view returns (uint256) {
        return _balances[_owner];
    }
function bug_intou7() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function allowance(address _owner, address _spender) public view returns (uint256) {
        return _allowed[_owner][_spender];
    }
function bug_intou23() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function transfer(address _to, uint256 _value) onlyPayloadSize(2 * 32) public returns (bool) {
        _transfer(msg.sender, _to, _value);
        return true;
    }
mapping(address => uint) balances_intou14;
function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);   
    balances_intou14[msg.sender] -= _value;   
    balances_intou14[_to] += _value;   
    return true;
  }
    function approve(address _spender, uint256 _value) public returns (bool) {
        _approve(msg.sender, _spender, _value);
        return true;
    }
mapping(address => uint) balances_intou30;
function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);   
    balances_intou30[msg.sender] -= _value;   
    balances_intou30[_to] += _value;   
    return true;
  }
    function transferFrom(address _from, address _to, uint256 _value) onlyPayloadSize(3 * 32) public returns (bool) {
        _transfer(_from, _to, _value);
        _approve(_from, msg.sender, _allowed[_from][msg.sender].sub(_value));
        return true;
    }
function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;    
}
    function _transfer(address _from, address _to, uint256 _value) internal {
        require(_to != address(0), "ERC20: transfer to the zero address");
        _balances[_from] = _balances[_from].sub(_value);
        _balances[_to] = _balances[_to].add(_value);
        emit Transfer(_from, _to, _value);
    }
function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function _approve(address _owner, address _spender, uint256 _value) internal {
        require(_owner != address(0), "ERC20: approve from the zero address");
        require(_spender != address(0), "ERC20: approve to the zero address");
        _allowed[_owner][_spender] = _value;
        emit Approval(_owner, _spender, _value);
    }
function bug_intou36(uint8 p_intou36) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou36;    
}
    function mint(address _to, uint256 _amount) onlyOwner public returns (bool) {
        require(_totalSupply.add(_amount) <= cap);
        _totalSupply = _totalSupply.add(_amount);
        _balances[_to] = _balances[_to].add(_amount);
        emit Mint(_to, _amount);
        emit Transfer(address(0), _to, _amount);
        return true;
    }
function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function transferBatch(address[] memory _tos, uint256[] memory _values) public returns (bool) {
        require(_tos.length == _values.length);
        for (uint256 i = 0; i < _tos.length; i++) {
            transfer(_tos[i], _values[i]);
        }
        return true;
    }
function bug_intou40(uint8 p_intou40) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou40;    
}
}
contract XLToken is TokenERC20 {
    constructor() TokenERC20(18*10**16, 12*10**16, "XL Token", "XL", 8) public {}
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