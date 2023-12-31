pragma solidity ^0.5.2;
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0);
        return a % b;
    }
}
interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function totalSupply() external view returns (uint256);
    function balanceOf(address who) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
contract ERC20 is IERC20 {
    using SafeMath for uint256;
    mapping (address => uint256) private _balances;
    mapping (address => mapping (address => uint256)) private _allowed;
    uint256 private _totalSupply;
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
    function balanceOf(address owner) public view returns (uint256) {
        return _balances[owner];
    }
    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowed[owner][spender];
    }
    function transfer(address to, uint256 value) public returns (bool) {
        _transfer(msg.sender, to, value);
        return true;
    }
    function approve(address spender, uint256 value) public returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }
    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        _transfer(from, to, value);
        _approve(from, msg.sender, _allowed[from][msg.sender].sub(value));
        return true;
    }
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowed[msg.sender][spender].add(addedValue));
        return true;
    }
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowed[msg.sender][spender].sub(subtractedValue));
        return true;
    }
    function _transfer(address from, address to, uint256 value) internal {
        require(to != address(0));
        _balances[from] = _balances[from].sub(value);
        _balances[to] = _balances[to].add(value);
        emit Transfer(from, to, value);
    }
    function _mint(address account, uint256 value) internal {
        require(account != address(0));
        _totalSupply = _totalSupply.add(value);
        _balances[account] = _balances[account].add(value);
        emit Transfer(address(0), account, value);
    }
    function _burn(address account, uint256 value) internal {
        require(account != address(0));
        _totalSupply = _totalSupply.sub(value);
        _balances[account] = _balances[account].sub(value);
        emit Transfer(account, address(0), value);
    }
    function _approve(address owner, address spender, uint256 value) internal {
        require(spender != address(0));
        require(owner != address(0));
        _allowed[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
    function _burnFrom(address account, uint256 value) internal {
        _burn(account, value);
        _approve(account, msg.sender, _allowed[account][msg.sender].sub(value));
    }
}
contract Controller {
    address private _owner;
    bool private _paused;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Paused(address account);
    event Unpaused(address account);
    mapping(address => bool) private owners;
    constructor() internal {
        setOwner(msg.sender);
    }
    modifier onlyOwner() {
        require(owners[msg.sender]);
        _;
    }
    function setOwner(address addr) internal returns(bool) {
        if (!owners[addr]) {
          owners[addr] = true;
          _owner = addr;
          return true; 
        }
    }
    function changeOwner(address newOwner) onlyOwner public returns(bool) {
        require (!owners[newOwner]);
          owners[newOwner];
          _owner = newOwner;
          emit OwnershipTransferred(_owner, newOwner);
          return true; 
        }
    function Owner() public view returns (address) {
        return _owner;
    }
    function paused() public view returns(bool) {
    return _paused;
    }
    modifier whenNotPaused() {
    require(!_paused);
    _;
    }
    modifier whenPaused() {
    require(_paused);
    _;
    }
    function pause() public onlyOwner whenNotPaused {
    _paused = true;
    emit Paused(msg.sender);
    }
    function unpause() public onlyOwner whenPaused {
    _paused = false;
    emit Unpaused(msg.sender);
    }
}
contract LobefyToken is ERC20, Controller {
    using SafeMath for uint256;
    string private _name = "Lobefy Token";
    string private _symbol = "CRWD";
    uint256 private _decimals = 18;
    address private team = 0xDA19316953D19f5f8C6361d68C6D0078c06285d3;
    uint256 private team1Balance = 50 * (10 ** 6) * (10 ** 18);
    constructor() public {
        mint(team, team1Balance);
    }
    function name() public view returns (string memory) {
        return _name;
    }
    function symbol() public view returns (string memory) {
        return _symbol;
    }
    function decimals() public view returns (uint256) {
        return _decimals;
    }
    function burn(uint256 value) public {
        _burn(msg.sender, value);
    }
    function burnFrom(address from, uint256 value) public {
        _burnFrom(from, value);
    }
    function mint(address to, uint256 value) public onlyOwner returns (bool) {
        _mint(to, value);
        return true;
    }
    function transfer(address to, uint256 value) public whenNotPaused returns (bool) {
        return super.transfer(to, value);
    }
    function transferFrom(address from, address to, uint256 value) public whenNotPaused returns (bool) {
        return super.transferFrom(from, to, value);
    }
    function approve(address spender, uint256 value) public whenNotPaused returns (bool) {
        return super.approve(spender, value);
    }
    function increaseAllowance(address spender, uint addedValue) public whenNotPaused returns (bool success) {
        return super.increaseAllowance(spender, addedValue);
    }
    function decreaseAllowance(address spender, uint subtractedValue) public whenNotPaused returns (bool success) {
        return super.decreaseAllowance(spender, subtractedValue);
    }
}