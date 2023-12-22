pragma solidity 0.4.21;
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}
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
contract ERC223 {
    function transfer(address _to, uint _value, bytes _data) public returns (bool success);
    function transfer(address _to, uint _value, bytes _data, string _fallback) public returns (bool success);
    event Transfer(address indexed from, address indexed to, uint value, bytes data);
}
contract ERC223ReceivingContract { 
    function tokenFallback(address _from, uint _value, bytes _data);
}
contract PGGamePlatform is ERC223, ERC20Basic {
  using SafeMath for uint256;
  string public name;
  string public symbol;
  uint8 public decimals;
  uint256 public totalSupply;
  mapping(address => uint256) public balances;
  function name() public view returns (string _name) {
    return name;
  }
  function symbol() public view returns (string _symbol) {
    return symbol;
  }
  function decimals() public view returns (uint8 _decimals) {
    return decimals;
  }
  function totalSupply() public view returns (uint256 _totalSupply) {
    return totalSupply;
  }
  function balanceOf(address _owner) public view returns (uint256 _balance) {
    return balances[_owner];
  }
  function PGGamePlatform() public{
      name = "PG Game Platform";
      symbol = "PGG";
      decimals = 4;
      totalSupply = 10000000000 * 10 ** uint(decimals);
      balances[msg.sender] = totalSupply;
  }
  function transfer(address _to, uint256 _value, bytes _data, string _fallback) public returns (bool _success) {
    if (isContract(_to)) {
      if (balanceOf(msg.sender) < _value)
      revert();
      balances[msg.sender] = balanceOf(msg.sender).sub(_value);
      balances[_to] = balanceOf(_to).add(_value);
      assert(_to.call.value(0)(bytes4(keccak256(_fallback)), msg.sender, _value, _data));
      Transfer(msg.sender, _to, _value, _data);
      return true;
    } else {
      return transferToAddress(_to, _value, _data);
    }
  }
  function transfer(address _to, uint256 _value, bytes _data) public returns (bool _success) {
    if (isContract(_to)) {
      return transferToContract(_to, _value, _data);
    } else {
      return transferToAddress(_to, _value, _data);
    }
  }
  function transfer(address _to, uint256 _value) public returns (bool _success) {
    bytes memory empty;
    if (isContract(_to)) {
      return transferToContract(_to, _value, empty);
    } else {
      return transferToAddress(_to, _value, empty);
    }
  }
  function isContract(address _addr) private view returns (bool _isContract) {
    uint length;
    assembly {
      length := extcodesize(_addr)
    }
    return (length > 0);
  }
  function transferToAddress(address _to, uint256 _value, bytes _data) private returns (bool _success) {
    if (balanceOf(msg.sender) < _value)
    revert();
    balances[msg.sender] = balanceOf(msg.sender).sub(_value);
    balances[_to] = balanceOf(_to).add(_value);
    Transfer(msg.sender, _to, _value, _data);
    return true;
  }
  function transferToContract(address _to, uint256 _value, bytes _data) private returns (bool _success) {
    if (balanceOf(msg.sender) < _value) {
        revert();
    }
    balances[msg.sender] = balanceOf(msg.sender).sub(_value);
    balances[_to] = balanceOf(_to).add(_value);
    ERC223ReceivingContract receiver = ERC223ReceivingContract(_to);
    receiver.tokenFallback(msg.sender, _value, _data);
    Transfer(msg.sender, _to, _value, _data);
    return true;
  }
}