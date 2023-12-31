pragma solidity ^0.4.16;
	pragma solidity ^0.4.20;
	pragma solidity ^0.4.20;
	contract Token {
	  function totalSupply () public view returns (uint256 supply);
	  function balanceOf (address _owner) public view returns (uint256 balance);
	  function transfer (address _to, uint256 _value)
	  public returns (bool success);
	  function transferFrom (address _from, address _to, uint256 _value)
	  public returns (bool success);
	  function approve (address _spender, uint256 _value)
	  public returns (bool success);
	  function allowance (address _owner, address _spender)
	  public view returns (uint256 remaining);
	  event Transfer (address indexed _from, address indexed _to, uint256 _value);
	  event Approval (
		address indexed _owner, address indexed _spender, uint256 _value);
	}
	pragma solidity ^0.4.20;
	contract SafeMath {
	  uint256 constant private MAX_UINT256 =
		0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
	  function safeAdd (uint256 x, uint256 y)
	  pure internal
	  returns (uint256 z) {
		assert (x <= MAX_UINT256 - y);
		return x + y;
	  }
	  function safeSub (uint256 x, uint256 y)
	  pure internal
	  returns (uint256 z) {
		assert (x >= y);
		return x - y;
	  }
	  function safeMul (uint256 x, uint256 y)
	  pure internal
	  returns (uint256 z) {
		if (y == 0) return 0;  
		assert (x <= MAX_UINT256 / y);
		return x * y;
	  }
	}
	contract AbstractToken is Token, SafeMath {
	  function AbstractToken () public {
	  }
	  function balanceOf (address _owner) public view returns (uint256 balance) {
		return accounts [_owner];
	  }
	  function transfer (address _to, uint256 _value)
	  public returns (bool success) {
		uint256 fromBalance = accounts [msg.sender];
		if (fromBalance < _value) return false;
		if (_value > 0 && msg.sender != _to) {
		  accounts [msg.sender] = safeSub (fromBalance, _value);
		  accounts [_to] = safeAdd (accounts [_to], _value);
		}
		Transfer (msg.sender, _to, _value);
		return true;
	  }
	  function transferFrom (address _from, address _to, uint256 _value)
	  public returns (bool success) {
		uint256 spenderAllowance = allowances [_from][msg.sender];
		if (spenderAllowance < _value) return false;
		uint256 fromBalance = accounts [_from];
		if (fromBalance < _value) return false;
		allowances [_from][msg.sender] =
		  safeSub (spenderAllowance, _value);
		if (_value > 0 && _from != _to) {
		  accounts [_from] = safeSub (fromBalance, _value);
		  accounts [_to] = safeAdd (accounts [_to], _value);
		}
		Transfer (_from, _to, _value);
		return true;
	  }
	  function approve (address _spender, uint256 _value)
	  public returns (bool success) {
		allowances [msg.sender][_spender] = _value;
		Approval (msg.sender, _spender, _value);
		return true;
	  }
	  function allowance (address _owner, address _spender)
	  public view returns (uint256 remaining) {
		return allowances [_owner][_spender];
	  }
	  mapping (address => uint256) internal accounts;
	  mapping (address => mapping (address => uint256)) internal allowances;
	}
	contract CuboToken is AbstractToken {
	  uint256 tokenCount;
	  function CuboToken (uint256 _tokenCount) public {
		tokenCount = _tokenCount;
		accounts [msg.sender] = _tokenCount;
	  }
	  function totalSupply () public view returns (uint256 supply) {
		return tokenCount;
	  }
	  function name () public pure returns (string result) {
		return "Cubo";
	  }
	  function symbol () public pure returns (string result) {
		return "CUBO";
	  }
	  function decimals () public pure returns (uint8 result) {
		return 6;
	  }
	  function approve (address _spender, uint256 _currentValue, uint256 _newValue)
		public returns (bool success) {
		if (allowance (msg.sender, _spender) == _currentValue)
		  return approve (_spender, _newValue);
		else return false;
	  }
	}