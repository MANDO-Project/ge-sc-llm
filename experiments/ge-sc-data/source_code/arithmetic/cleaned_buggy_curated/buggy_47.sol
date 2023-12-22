pragma solidity ^0.5.11;
contract ERC20Interface {
    function totalSupply() public view returns (uint);
mapping(address => uint) public lockTime_intou17;
function increaseLockTime_intou17(uint _secondsToIncrease) public {
        lockTime_intou17[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou17() public {
        require(now > lockTime_intou17[msg.sender]);    
        uint transferValue_intou17 = 10;           
        msg.sender.transfer(transferValue_intou17);
    }
    function balanceOf(address tokenOwner) public view returns (uint balance);
mapping(address => uint) public lockTime_intou37;
function increaseLockTime_intou37(uint _secondsToIncrease) public {
        lockTime_intou37[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou37() public {
        require(now > lockTime_intou37[msg.sender]);    
        uint transferValue_intou37 = 10;           
        msg.sender.transfer(transferValue_intou37);
    }
    function transfer(address to, uint tokens) public returns (bool success);
function bug_intou3() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function allowance(address tokenOwner, address spender) public view returns (uint remaining);
mapping(address => uint) public lockTime_intou9;
function increaseLockTime_intou9(uint _secondsToIncrease) public {
        lockTime_intou9[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou9() public {
        require(now > lockTime_intou9[msg.sender]);    
        uint transferValue_intou9 = 10;           
        msg.sender.transfer(transferValue_intou9);
    }
    function approve(address spender, uint tokens) public returns (bool success);
mapping(address => uint) public lockTime_intou25;
function increaseLockTime_intou25(uint _secondsToIncrease) public {
        lockTime_intou25[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou25() public {
        require(now > lockTime_intou25[msg.sender]);    
        uint transferValue_intou25 = 10;           
        msg.sender.transfer(transferValue_intou25);
    }
    function transferFrom(address from, address to, uint tokens) public returns (bool success);
function bug_intou19() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}
contract AcunarToken is ERC20Interface{
  mapping(address => uint) balances_intou18;
function transfer_intou18(address _to, uint _value) public returns (bool) {
    require(balances_intou18[msg.sender] - _value >= 0);   
    balances_intou18[msg.sender] -= _value;   
    balances_intou18[_to] += _value;   
    return true;
  }
  string public name = "Acunar";
  mapping(address => uint) public lockTime_intou29;
function increaseLockTime_intou29(uint _secondsToIncrease) public {
        lockTime_intou29[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou29() public {
        require(now > lockTime_intou29[msg.sender]);    
        uint transferValue_intou29 = 10;           
        msg.sender.transfer(transferValue_intou29);
    }
  string public symbol = "ACN";
  mapping(address => uint) balances_intou6;
function transfer_intou62(address _to, uint _value) public returns (bool) {
    require(balances_intou6[msg.sender] - _value >= 0);   
    balances_intou6[msg.sender] -= _value;   
    balances_intou6[_to] += _value;   
    return true;
  }
  uint public decimals = 0;
  function bug_intou16(uint8 p_intou16) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou16;    
}
  uint public supply;
  function bug_intou24(uint8 p_intou24) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou24;    
}
  address public founder;
  mapping(address => uint) public lockTime_intou5;
function increaseLockTime_intou5(uint _secondsToIncrease) public {
        lockTime_intou5[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou5() public {
        require(now > lockTime_intou5[msg.sender]);    
        uint transferValue_intou5 = 10;           
        msg.sender.transfer(transferValue_intou5);
    }
  mapping(address => uint) public balances;
  function bug_intou15() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  mapping(address => mapping(address => uint)) allowed;
    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
    constructor() public{
        supply = 200000000;
        founder = msg.sender;
        balances[founder] = supply;
    }
mapping(address => uint) balances_intou26;
function transfer_intou26(address _to, uint _value) public returns (bool) {
    require(balances_intou26[msg.sender] - _value >= 0);   
    balances_intou26[msg.sender] -= _value;   
    balances_intou26[_to] += _value;   
    return true;
  }
    function allowance(address tokenOwner, address spender) view public returns(uint){
        return allowed[tokenOwner][spender];
    }
function bug_intou20(uint8 p_intou20) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou20;    
}
    function approve(address spender, uint tokens) public returns(bool){
        require(balances[msg.sender] >= tokens);
        require(tokens > 0);
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        return true;
    }
function bug_intou32(uint8 p_intou32) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou32;    
}
    function transferFrom(address from, address to, uint tokens) public returns(bool){
        require(allowed[from][to] >= tokens);
        require(balances[from] >= tokens);
        balances[from] -= tokens;
        balances[to] += tokens;
        allowed[from][to] -= tokens;
        return true;
    }
mapping(address => uint) balances_intou38;
function transfer_intou38(address _to, uint _value) public returns (bool) {
    require(balances_intou38[msg.sender] - _value >= 0);   
    balances_intou38[msg.sender] -= _value;   
    balances_intou38[_to] += _value;   
    return true;
  }
    function totalSupply() public view returns (uint){
        return supply;
    }
function bug_intou4(uint8 p_intou4) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou4;    
}
    function balanceOf(address tokenOwner) public view returns (uint balance){
         return balances[tokenOwner];
     }
function bug_intou7() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function transfer(address to, uint tokens) public returns (bool success){
         require(balances[msg.sender] >= tokens && tokens > 0);
         balances[to] += tokens;
         balances[msg.sender] -= tokens;
         emit Transfer(msg.sender, to, tokens);
         return true;
     }
function bug_intou23() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
}
contract AcunarIEO is AcunarToken{
  function bug_intou28(uint8 p_intou28) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou28;    
}
  address public admin;
  mapping(address => uint) balances_intou34;
function transfer_intou34(address _to, uint _value) public returns (bool) {
    require(balances_intou34[msg.sender] - _value >= 0);   
    balances_intou34[msg.sender] -= _value;   
    balances_intou34[_to] += _value;   
    return true;
  }
  address payable public deposit;
  mapping(address => uint) public lockTime_intou21;
function increaseLockTime_intou21(uint _secondsToIncrease) public {
        lockTime_intou21[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou21() public {
        require(now > lockTime_intou21[msg.sender]);    
        uint transferValue_intou21 = 10;           
        msg.sender.transfer(transferValue_intou21);
    }
  uint tokenPrice = 0.0001 ether;
  mapping(address => uint) balances_intou10;
function transfer_intou10(address _to, uint _value) public returns (bool) {
    require(balances_intou10[msg.sender] - _value >= 0);   
    balances_intou10[msg.sender] -= _value;   
    balances_intou10[_to] += _value;   
    return true;
  }
  uint public hardCap =21000 ether;
  mapping(address => uint) balances_intou22;
function transfer_intou22(address _to, uint _value) public returns (bool) {
    require(balances_intou22[msg.sender] - _value >= 0);   
    balances_intou22[msg.sender] -= _value;   
    balances_intou22[_to] += _value;   
    return true;
  }
  uint public raisedAmount;
  function bug_intou12(uint8 p_intou12) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou12;    
}
  uint public saleStart = now;
    uint public saleEnd = now + 14515200;  
    uint public coinTradeStart = saleEnd + 15120000;  
  function bug_intou11() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
  uint public maxInvestment = 30 ether;
  mapping(address => uint) public lockTime_intou1;
function increaseLockTime_intou1(uint _secondsToIncrease) public {
        lockTime_intou1[msg.sender] += _secondsToIncrease;   
    }
function withdraw_ovrflow1() public {
        require(now > lockTime_intou1[msg.sender]);    
        uint transferValue_intou1 = 10;           
        msg.sender.transfer(transferValue_intou1);
    }
  uint public minInvestment = 0.1 ether;
    enum State { beforeStart, running, afterEnd, halted}
  mapping(address => uint) balances_intou2;
function transfer_undrflow2(address _to, uint _value) public returns (bool) {
    require(balances_intou2[msg.sender] - _value >= 0);   
    balances_intou2[msg.sender] -= _value;   
    balances_intou2[_to] += _value;   
    return true;
  }
  State public ieoState;
    modifier onlyAdmin(){
        require(msg.sender == admin);
        _;
    }
  mapping(address => uint) public lockTime_intou13;
function increaseLockTime_intou13(uint _secondsToIncrease) public {
        lockTime_intou13[msg.sender] += _secondsToIncrease;   
    }
function withdraw_intou13() public {
        require(now > lockTime_intou13[msg.sender]);    
        uint transferValue_intou13 = 10;           
        msg.sender.transfer(transferValue_intou13);
    }
  event Invest(address investor, uint value, uint tokens);
    constructor(address payable _deposit) public{
        deposit = _deposit;
        admin = msg.sender;
        ieoState = State.beforeStart;
    }
mapping(address => uint) balances_intou14;
function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);   
    balances_intou14[msg.sender] -= _value;   
    balances_intou14[_to] += _value;   
    return true;
  }
    function halt() public onlyAdmin{
        ieoState = State.halted;
    }
mapping(address => uint) balances_intou30;
function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);   
    balances_intou30[msg.sender] -= _value;   
    balances_intou30[_to] += _value;   
    return true;
  }
    function unhalt() public onlyAdmin{
        ieoState = State.running;
    }
function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;    
}
    function changeDepositAddress(address payable newDeposit) public onlyAdmin{
        deposit = newDeposit;
    }
function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function getCurrentState() public view returns(State){
        if(ieoState == State.halted){
            return State.halted;
        }else if(block.timestamp < saleStart){
            return State.beforeStart;
        }else if(block.timestamp >= saleStart && block.timestamp <= saleEnd){
            return State.running;
        }else{
            return State.afterEnd;
        }
    }
function bug_intou36(uint8 p_intou36) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou36;    
}
    function invest() payable public returns(bool){
        ieoState = getCurrentState();
        require(ieoState == State.running);
        require(msg.value >= minInvestment && msg.value <= maxInvestment);
        uint tokens = msg.value / tokenPrice;
        require(raisedAmount + msg.value <= hardCap);
        raisedAmount += msg.value;
        balances[msg.sender] += tokens;
        balances[founder] -= tokens;
        deposit.transfer(msg.value); 
        emit Invest(msg.sender, msg.value, tokens);
        return true;
    }
function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function () payable external{
        invest();
    }
function bug_intou40(uint8 p_intou40) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou40;    
}
    function burn() public returns(bool){
        ieoState = getCurrentState();
        require(ieoState == State.afterEnd);
        balances[founder] = 0;
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
    function transfer(address to, uint value) public returns(bool){
        require(block.timestamp > coinTradeStart);
        super.transfer(to, value);
    }
function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
    function transferFrom(address _from, address _to, uint _value) public returns(bool){
        require(block.timestamp > coinTradeStart);
        super.transferFrom(_from, _to, _value);
    }
function bug_intou31() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;    
}
}