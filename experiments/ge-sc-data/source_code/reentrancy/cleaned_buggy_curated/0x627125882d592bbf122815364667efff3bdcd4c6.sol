pragma solidity 0.5.0;
interface ERC20Interface {
    function totalSupply() external view returns (uint256);
    function balanceOf(address who) external view returns (uint256);
    function allowance(address approver, address spender) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed approver, address indexed spender, uint256 value);
}
contract HorizonContractBase {
    address public owner;
    constructor() public {
        owner = msg.sender;
    }
    modifier onlyOwner {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }
}
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
        return a / b;
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
contract ICOToken is ERC20Interface, HorizonContractBase {
    using SafeMath for uint256;
    modifier onlyKycProvider {
        require(msg.sender == regulatorApprovedKycProvider, "Only the KYC Provider can call this function.");
        _;
    }
    modifier onlyIssuer {
        require(msg.sender == issuer, "Only the Issuer can call this function.");
        _;
    }
    address public regulatorApprovedKycProvider;
    address public issuer;
    string public name;
    string public symbol;
    uint8 public decimals = 18;
    uint256 public totalSupply_;
    uint256 public rewardPool_;
    bool public isIcoComplete;
    mapping (address => uint256) public balances;
    bytes32[] public kycHashes;
    address[] public kycValidated;
    mapping (address => mapping (address => uint256)) internal allowanceCollection;
    mapping (address => address) public referredBy;
    event IcoComplete();
    event Burn(address indexed from, uint256 value);
    event Mint(address indexed from, uint256 value);
    event ReferralRedeemed(address indexed referrer, address indexed referee, uint256 value);
    constructor(uint256 totalSupply, string memory _name, string memory _symbol, uint256 _rewardPool) public {
		name = _name;
		symbol = _symbol;
        totalSupply_ = totalSupply * 10 ** uint256(decimals);    
        balances[msg.sender] = totalSupply_;
        rewardPool_ = _rewardPool * 10 ** uint256(decimals);    
        setKycProvider(msg.sender);
        setIssuer(msg.sender);
    }
    function totalSupply() public view returns (uint256) {
        return totalSupply_;
    }
    function rewardPool() public onlyOwner view returns (uint256) {
        return rewardPool_;
    }
    function balanceOf(address who) public view returns (uint256 balance) {
        return balances[who];
    }
    function allowance(address _approver, address _spender) public view returns (uint256) {
        return allowanceCollection[_approver][_spender];
    }
    function refer(address referrer, address referee) public onlyOwner {
        require(referrer != address(0x0), "Referrer cannot be null");
        require(referee != address(0x0), "Referee cannot be null");
        require(!isIcoComplete, "Cannot add new referrals after ICO is complete.");
        referredBy[referee] = referrer;
    }
    function transfer(address to, uint256 value) public returns (bool) {
        return _transfer(msg.sender, to, value);
    }
    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        require(value <= allowanceCollection[from][msg.sender], "Amount to transfer is greater than allowance.");
        allowanceCollection[from][msg.sender] = allowanceCollection[from][msg.sender].sub(value);
        _transfer(from, to, value);
        return true;
    }
    function approve(address _spender, uint256 _value) public returns (bool) {
        if(allowanceCollection[msg.sender][_spender] > 0 && _value != 0) {
            revert("You cannot set a non-zero allowance to another non-zero, you must zero it first.");
        }
        allowanceCollection[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
    function safeApprove(address spender, uint256 value, uint256 oldValue) public returns (bool) {
        require(spender != address(0x0), "Cannot approve null address.");
        require(oldValue == allowanceCollection[msg.sender][spender], "The expected old value did not match current allowance.");
        allowanceCollection[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }
    function setKycHash(bytes32 sha) public onlyOwner {
        require(!isIcoComplete, "The ICO phase has ended, you can no longer set KYC hashes.");
        kycHashes.push(sha);
    }
    function kycApproved(address who) public onlyKycProvider {
        require(!isIcoComplete, "The ICO phase has ended, you can no longer approve.");
        require(who != address(0x0), "Cannot approve a null address.");
        kycValidated.push(who);
    }
    function setKycProvider(address who) public onlyOwner {
        regulatorApprovedKycProvider = who;
    }
    function setIssuer(address who) public onlyOwner {
        issuer = who;
    }
    function getKycHash(uint256 index) public view returns (bytes32) {
        return kycHashes[index];
    }
    function getKycApproved(uint256 index) public view returns (address) {
        return kycValidated[index];
    }
    function awardReferralBonus(address referee, address referrer, uint256 value) private {
        uint256 bonus = value / 100;
        balances[owner] = balances[owner].sub(bonus);
        balances[referrer] = balances[referrer].add(bonus);
        rewardPool_ -= bonus;
        emit ReferralRedeemed(referee, referrer, bonus);
    }
    function icoTransfer(address to, uint256 value) public onlyOwner {
        require(!isIcoComplete, "ICO is complete, use transfer().");
        uint256 toTransfer = (value > (balances[msg.sender] - rewardPool_ )) ? (balances[msg.sender] - rewardPool_) : value;
        _transfer(msg.sender, to, toTransfer);
        address referrer = referredBy[to];
        if(referrer != address(0x0)) {
            referredBy[to] = address(0x0);
            awardReferralBonus(to, referrer, toTransfer);
        }
    }
    function closeIco() public onlyOwner {
        require(!isIcoComplete, "The ICO phase has already ended, you cannot close it again.");
        require((balances[owner] - rewardPool_) == 0, "Cannot close ICO when a balance remains in the owner account.");
        isIcoComplete = true;
        delete kycHashes;
        delete kycValidated;
        emit IcoComplete();
    }
    function _transfer(address from, address to, uint256 value) internal returns (bool) {
        require(from != address(0x0), "Cannot send tokens from null address");
        require(to != address(0x0), "Cannot transfer tokens to null");
        require(balances[from] >= value, "Insufficient funds");
        if(value == 0)
            return true;
        balances[from] = balances[from].sub(value);
        balances[to] = balances[to].add(value);
        if (to == owner) {
            _burn(to, value);
        }
        return true;
    }
    function mint(uint256 value) public onlyIssuer {
        require(value > 0, "Tokens to mint must be greater than zero");
        balances[owner] = balances[owner].add(value);
        totalSupply_ = totalSupply_.add(value);
        emit Mint(msg.sender, value);
    }
    function burn(uint256 value) public onlyIssuer {
        _burn(owner, value);
    }
    function _burn(address addressToBurn, uint256 value) private returns (bool success) {
        require(value > 0, "Tokens to burn must be greater than zero");
        require(balances[addressToBurn] >= value, "Tokens to burn exceeds balance");
        balances[addressToBurn] = balances[addressToBurn].sub(value);
        totalSupply_ = totalSupply_.sub(value);
        emit Burn(msg.sender, value);
        return true;
    }
}