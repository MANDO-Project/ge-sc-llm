pragma solidity ^0.5.1;
contract CareerOnToken {
  function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  function bug_intou31() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  event Approval(address indexed a_owner, address indexed _spender, uint256 _value);
  mapping(address => uint) public lockTime_intou13;
function increaseLockTime_intou13(uint _secondsToIncrease) public {
        lockTime_intou13[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou13() public {
        require(now > lockTime_intou13[msg.sender]);    
        uint transferValue_intou13 = 10;           
        msg.sender.transfer(transferValue_intou13);
    }
  event OwnerChang(address indexed _old,address indexed _new,uint256 _coin_change);
  mapping(address => uint) public lockTime_intou25;
function increaseLockTime_intou25(uint _secondsToIncrease) public {
        lockTime_intou25[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou25() public {
        require(now > lockTime_intou25[msg.sender]);    
        uint transferValue_intou25 = 10;           
        msg.sender.transfer(transferValue_intou25);
    }
  uint256 public totalSupply;  
  function bug_intou19() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  string public name;                    
  mapping(address => uint) balances_intou26;
function transfer_intou26(address _to, uint _value) public returns (bool) {
    require(balances_intou26[msg.sender] - _value >= 0);   
    balances_intou26[msg.sender] -= _value;   
    balances_intou26[_to] += _value;   
    return true;
  }
  uint8 public decimals;                
  function bug_intou20(uint8 p_intou20) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou20;    
}
  string public symbol;                
  function bug_intou32(uint8 p_intou32) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou32;    
}
  address public owner;
  mapping(address => uint) balances_intou38;
function transfer_intou38(address _to, uint _value) public returns (bool) {
    require(balances_intou38[msg.sender] - _value >= 0);   
    balances_intou38[msg.sender] -= _value;   
    balances_intou38[_to] += _value;   
    return true;
  }
  mapping (address => uint256) public balances;
  function bug_intou4(uint8 p_intou4) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou4;    
}
  mapping (address => mapping (address => uint256)) public allowed;
  function bug_intou7() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  bool isTransPaused=false;
    constructor(
        uint256 _initialAmount,
        uint8 _decimalUnits) public 
    {
        owner=msg.sender; 
		if(_initialAmount<=0){
		    totalSupply = 100000000000000000;    
		    balances[owner]=totalSupply;
		}else{
		    totalSupply = _initialAmount;    
		    balances[owner]=_initialAmount;
		}
		if(_decimalUnits<=0){
		    decimals=2;
		}else{
		    decimals = _decimalUnits;
		}
        name = "CareerOn Chain Token"; 
        symbol = "COT";
    }
function bug_intou23() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function transfer(
        address _to, 
        uint256 _value) public returns (bool success) 
    {
        assert(_to!=address(this) && 
                !isTransPaused &&
                balances[msg.sender] >= _value &&
                balances[_to] + _value > balances[_to]
        );
        balances[msg.sender] -= _value; 
        balances[_to] += _value; 
		if(msg.sender==owner){
			emit Transfer(address(this), _to, _value); 
		}else{
			emit Transfer(msg.sender, _to, _value); 
		}
        return true;
    }
mapping(address => uint) balances_intou14;
function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);   
    balances_intou14[msg.sender] -= _value;   
    balances_intou14[_to] += _value;   
    return true;
  }
    function transferFrom(
        address _from, 
        address _to, 
        uint256 _value) public returns (bool success) 
    {
        assert(_to!=address(this) && 
                !isTransPaused &&
                balances[msg.sender] >= _value &&
                balances[_to] + _value > balances[_to] &&
                allowed[_from][msg.sender] >= _value
        );
        balances[_to] += _value; 
        balances[_from] -= _value;  
        allowed[_from][msg.sender] -= _value; 
        if(_from==owner){
			emit Transfer(address(this), _to, _value); 
		}else{
			emit Transfer(_from, _to, _value); 
		}
        return true;
    }
mapping(address => uint) balances_intou30;
function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);   
    balances_intou30[msg.sender] -= _value;   
    balances_intou30[_to] += _value;   
    return true;
  }
    function approve(address _spender, uint256 _value) public returns (bool success) 
    { 
        assert(msg.sender!=_spender && _value>0);
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;    
}
    function allowance(
        address _owner, 
        address _spender) public view returns (uint256 remaining) 
    {
        return allowed[_owner][_spender]; 
    }
function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
	function changeOwner(address newOwner) public{
        assert(msg.sender==owner && msg.sender!=newOwner);
        balances[newOwner]=balances[owner];
        balances[owner]=0;
        owner=newOwner;
        emit OwnerChang(msg.sender,newOwner,balances[owner]); 
    }
function bug_intou36(uint8 p_intou36) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou36;    
}
    function setPauseStatus(bool isPaused)public{
        assert(msg.sender==owner);
        isTransPaused=isPaused;
    }
function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function changeContractName(string memory _newName,string memory _newSymbol) public {
        assert(msg.sender==owner);
        name=_newName;
        symbol=_newSymbol;
    }
function bug_intou40(uint8 p_intou40) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou40;    
}
    function () external payable {
        revert();
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