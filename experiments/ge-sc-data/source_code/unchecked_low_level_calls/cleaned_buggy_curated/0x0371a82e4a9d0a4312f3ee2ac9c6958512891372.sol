contract ERC20Basic {
  uint256 public totalSupply;
  function balanceOf(address who) constant returns (uint256);
  function transfer(address to, uint256 value) returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) constant returns (uint256);
  function transferFrom(address from, address to, uint256 value) returns (bool);
  function approve(address spender, uint256 value) returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}
contract SafeMath {
  function safeMul(uint a, uint b) internal returns (uint) {
    uint c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }
  function safeDiv(uint a, uint b) internal returns (uint) {
    assert(b > 0);
    uint c = a / b;
    assert(a == b * c + a % b);
    return c;
  }
  function safeSub(uint a, uint b) internal returns (uint) {
    assert(b <= a);
    return a - b;
  }
  function safeAdd(uint a, uint b) internal returns (uint) {
    uint c = a + b;
    assert(c>=a && c>=b);
    return c;
  }
  function max64(uint64 a, uint64 b) internal constant returns (uint64) {
    return a >= b ? a : b;
  }
  function min64(uint64 a, uint64 b) internal constant returns (uint64) {
    return a < b ? a : b;
  }
  function max256(uint256 a, uint256 b) internal constant returns (uint256) {
    return a >= b ? a : b;
  }
  function min256(uint256 a, uint256 b) internal constant returns (uint256) {
    return a < b ? a : b;
  }
}
contract StandardToken is ERC20, SafeMath {
  event Minted(address receiver, uint amount);
  mapping(address => uint) balances;
  mapping (address => mapping (address => uint)) allowed;
  function isToken() public constant returns (bool weAre) {
    return true;
  }
  function transfer(address _to, uint _value) returns (bool success) {
    balances[msg.sender] = safeSub(balances[msg.sender], _value);
    balances[_to] = safeAdd(balances[_to], _value);
    Transfer(msg.sender, _to, _value);
    return true;
  }
  function transferFrom(address _from, address _to, uint _value) returns (bool success) {
    uint _allowance = allowed[_from][msg.sender];
    balances[_to] = safeAdd(balances[_to], _value);
    balances[_from] = safeSub(balances[_from], _value);
    allowed[_from][msg.sender] = safeSub(_allowance, _value);
    Transfer(_from, _to, _value);
    return true;
  }
  function balanceOf(address _owner) constant returns (uint balance) {
    return balances[_owner];
  }
  function approve(address _spender, uint _value) returns (bool success) {
    if ((_value != 0) && (allowed[msg.sender][_spender] != 0)) throw;
    allowed[msg.sender][_spender] = _value;
    Approval(msg.sender, _spender, _value);
    return true;
  }
  function allowance(address _owner, address _spender) constant returns (uint remaining) {
    return allowed[_owner][_spender];
  }
}
contract BurnableToken is StandardToken {
  address public constant BURN_ADDRESS = 0;
  event Burned(address burner, uint burnedAmount);
  function burn(uint burnAmount) {
    address burner = msg.sender;
    balances[burner] = safeSub(balances[burner], burnAmount);
    totalSupply = safeSub(totalSupply, burnAmount);
    Burned(burner, burnAmount);
  }
}
contract UpgradeAgent {
  uint public originalSupply;
  function isUpgradeAgent() public constant returns (bool) {
    return true;
  }
  function upgradeFrom(address _from, uint256 _value) public;
}
contract UpgradeableToken is StandardToken {
  address public upgradeMaster;
  UpgradeAgent public upgradeAgent;
  uint256 public totalUpgraded;
  enum UpgradeState {Unknown, NotAllowed, WaitingForAgent, ReadyToUpgrade, Upgrading}
  event Upgrade(address indexed _from, address indexed _to, uint256 _value);
  event UpgradeAgentSet(address agent);
  function UpgradeableToken(address _upgradeMaster) {
    upgradeMaster = _upgradeMaster;
  }
  function upgrade(uint256 value) public {
      UpgradeState state = getUpgradeState();
      if(!(state == UpgradeState.ReadyToUpgrade || state == UpgradeState.Upgrading)) {
        throw;
      }
      if (value == 0) throw;
      balances[msg.sender] = safeSub(balances[msg.sender], value);
      totalSupply = safeSub(totalSupply, value);
      totalUpgraded = safeAdd(totalUpgraded, value);
      upgradeAgent.upgradeFrom(msg.sender, value);
      Upgrade(msg.sender, upgradeAgent, value);
  }
  function setUpgradeAgent(address agent) external {
      if(!canUpgrade()) {
        throw;
      }
      if (agent == 0x0) throw;
      if (msg.sender != upgradeMaster) throw;
      if (getUpgradeState() == UpgradeState.Upgrading) throw;
      upgradeAgent = UpgradeAgent(agent);
      if(!upgradeAgent.isUpgradeAgent()) throw;
      if (upgradeAgent.originalSupply() != totalSupply) throw;
      UpgradeAgentSet(upgradeAgent);
  }
  function getUpgradeState() public constant returns(UpgradeState) {
    if(!canUpgrade()) return UpgradeState.NotAllowed;
    else if(address(upgradeAgent) == 0x00) return UpgradeState.WaitingForAgent;
    else if(totalUpgraded == 0) return UpgradeState.ReadyToUpgrade;
    else return UpgradeState.Upgrading;
  }
  function setUpgradeMaster(address master) public {
      if (master == 0x0) throw;
      if (msg.sender != upgradeMaster) throw;
      upgradeMaster = master;
  }
  function canUpgrade() public constant returns(bool) {
     return true;
  }
}
contract Ownable {
  address public owner;
  function Ownable() {
    owner = msg.sender;
  }
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
  function transferOwnership(address newOwner) onlyOwner {
    require(newOwner != address(0));      
    owner = newOwner;
  }
}
contract ReleasableToken is ERC20, Ownable {
  address public releaseAgent;
  bool public released = false;
  mapping (address => bool) public transferAgents;
  modifier canTransfer(address _sender) {
    if(!released) {
        if(!transferAgents[_sender]) {
            throw;
        }
    }
    _;
  }
  function setReleaseAgent(address addr) onlyOwner inReleaseState(false) public {
    releaseAgent = addr;
  }
  function setTransferAgent(address addr, bool state) onlyOwner inReleaseState(false) public {
    transferAgents[addr] = state;
  }
  function releaseTokenTransfer() public onlyReleaseAgent {
    released = true;
  }
  modifier inReleaseState(bool releaseState) {
    if(releaseState != released) {
        throw;
    }
    _;
  }
  modifier onlyReleaseAgent() {
    if(msg.sender != releaseAgent) {
        throw;
    }
    _;
  }
  function transfer(address _to, uint _value) canTransfer(msg.sender) returns (bool success) {
   return super.transfer(_to, _value);
  }
  function transferFrom(address _from, address _to, uint _value) canTransfer(_from) returns (bool success) {
    return super.transferFrom(_from, _to, _value);
  }
}
library SafeMathLib {
  function times(uint a, uint b) returns (uint) {
    uint c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }
  function minus(uint a, uint b) returns (uint) {
    assert(b <= a);
    return a - b;
  }
  function plus(uint a, uint b) returns (uint) {
    uint c = a + b;
    assert(c>=a);
    return c;
  }
}
contract MintableToken is StandardToken, Ownable {
  using SafeMathLib for uint;
  bool public mintingFinished = false;
  mapping (address => bool) public mintAgents;
  event MintingAgentChanged(address addr, bool state  );
  function mint(address receiver, uint amount) onlyMintAgent canMint public {
    totalSupply = totalSupply.plus(amount);
    balances[receiver] = balances[receiver].plus(amount);
    Transfer(0, receiver, amount);
  }
  function setMintAgent(address addr, bool state) onlyOwner canMint public {
    mintAgents[addr] = state;
    MintingAgentChanged(addr, state);
  }
  modifier onlyMintAgent() {
    if(!mintAgents[msg.sender]) {
        throw;
    }
    _;
  }
  modifier canMint() {
    if(mintingFinished) throw;
    _;
  }
}
contract CrowdsaleToken is ReleasableToken, MintableToken, UpgradeableToken {
  event UpdatedTokenInformation(string newName, string newSymbol);
  string public name;
  string public symbol;
  uint public decimals;
  function CrowdsaleToken(string _name, string _symbol, uint _initialSupply, uint _decimals, bool _mintable)
    UpgradeableToken(msg.sender) {
    owner = msg.sender;
    name = _name;
    symbol = _symbol;
    totalSupply = _initialSupply;
    decimals = _decimals;
    balances[owner] = totalSupply;
    if(totalSupply > 0) {
      Minted(owner, totalSupply);
    }
    if(!_mintable) {
      mintingFinished = true;
      if(totalSupply == 0) {
        throw;  
      }
    }
  }
  function releaseTokenTransfer() public onlyReleaseAgent {
    mintingFinished = true;
    super.releaseTokenTransfer();
  }
  function canUpgrade() public constant returns(bool) {
    return released && super.canUpgrade();
  }
  function setTokenInformation(string _name, string _symbol) onlyOwner {
    name = _name;
    symbol = _symbol;
    UpdatedTokenInformation(name, symbol);
  }
}
contract BurnableCrowdsaleToken is BurnableToken, CrowdsaleToken {
  function BurnableCrowdsaleToken(string _name, string _symbol, uint _initialSupply, uint _decimals, bool _mintable)
    CrowdsaleToken(_name, _symbol, _initialSupply, _decimals, _mintable) {
  }
}