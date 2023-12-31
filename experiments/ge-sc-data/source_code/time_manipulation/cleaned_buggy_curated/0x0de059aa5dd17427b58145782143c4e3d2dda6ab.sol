pragma solidity 0.4.24;
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        assert(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);
        return c;
    }
}
contract ERC20Basic {
    function totalSupply() public view returns (uint256);
    function balanceOf(address who) public view returns (uint256);
    function transfer(address to, uint256 value) public returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
}
contract ERC20 is ERC20Basic {
    function allowance(address owner, address spender) public view returns (uint256);
    function transferFrom(address from, address to, uint256 value) public returns (bool);
    function approve(address spender, uint256 value) public returns (bool);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
contract BasicToken is ERC20Basic {
    using SafeMath for uint256;
    mapping(address => uint256) balances;
    uint256 public totalSupply_;
    function totalSupply() public view returns (uint256) {
        return totalSupply_;
    }
    function transfer(address _to, uint256 _value) public returns (bool) {
        require(msg.data.length>=(2*32)+4);
        require(_to != address(0));
        require(_value <= balances[msg.sender]);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer (msg.sender, _to, _value);
        return true;
    }
    function balanceOf(address _owner) public view returns (uint256 balance) {
        return balances[_owner];
    }
}
contract StandardToken is ERC20, BasicToken {
    mapping (address => mapping (address => uint256)) internal allowed;
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(_to != address(0));
        require(_value <= balances[_from]);
        require(_value <= allowed[_from][msg.sender]);
        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
        emit Transfer(_from, _to, _value);
        return true;
    }
    function approve(address _spender, uint256 _value) public returns (bool) {
        require(_value==0||allowed[msg.sender][_spender]==0);
        require(msg.data.length>=(2*32)+4);
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
    function allowance(address _owner, address _spender) public view returns (uint256) {
        return allowed[_owner][_spender];
    }
    function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
        allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
        emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }
    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
        uint oldValue = allowed[msg.sender][_spender];
        if (_subtractedValue > oldValue) {
            allowed[msg.sender][_spender] = 0;
        } else {
            allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
        }
        emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }
}
contract CodeToken is StandardToken {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public precentDecimal=2;
    uint256 public minerAndDeveloperPercent=70;
    uint256 public openDaoFundPercent=10;
    uint256 public codeCoinCoreTeamPercent=10;
    uint256 public mineralcloudFundPercent=10;
    address public minerAndDeveloperFundAccount;
    address public openDaoFundAccount;
    address public codeCoinCoreTeamAccount;
    address public mineralcloudFundAccount;
    uint256 public minerAndDeveloperFundBalnace;
    uint256 public openDaoFundBalnace;
    uint256 public codeCoinCoreTeamBalnace;
    uint256 public mineralcloudFundBalnace;
    constructor(string _name,string _symbol, uint8 _decimals, uint256 _initialSupply,
        address _minerAndDeveloperFundAccount,address _openDaoFundAccount,address _codeCoinCoreTeamAccount,address _mineralcloudFundAccount) public {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
        totalSupply_ = _initialSupply*10**uint256(_decimals);
        minerAndDeveloperFundAccount=_minerAndDeveloperFundAccount;
        openDaoFundAccount=_openDaoFundAccount;
        codeCoinCoreTeamAccount=_codeCoinCoreTeamAccount;
        mineralcloudFundAccount=_mineralcloudFundAccount;
        minerAndDeveloperFundBalnace=totalSupply_.mul(minerAndDeveloperPercent).div(10 ** precentDecimal);
        openDaoFundBalnace=totalSupply_.mul(openDaoFundPercent).div(10 ** precentDecimal);
        codeCoinCoreTeamBalnace=totalSupply_.mul(codeCoinCoreTeamPercent).div(10 ** precentDecimal);
        mineralcloudFundBalnace=totalSupply_.mul(mineralcloudFundPercent).div(10 ** precentDecimal);
        balances[_minerAndDeveloperFundAccount]=minerAndDeveloperFundBalnace;
        balances[_openDaoFundAccount]=openDaoFundBalnace;
        balances[_codeCoinCoreTeamAccount]=codeCoinCoreTeamBalnace;
        balances[_mineralcloudFundAccount]=mineralcloudFundBalnace;
    }
    function transfer(address _to, uint256 _value) public returns (bool) {
       return super.transfer(_to, _value);
    } 
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        return super.transferFrom(_from, _to, _value);
    }
     function() public payable{
         revert();
     }
}