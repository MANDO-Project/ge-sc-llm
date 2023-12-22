pragma solidity >=0.5.11;
interface IERC20 {
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}
interface Marmo {
    function signer() external view returns (address _signer);
}
library ECDSA {
    function recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        if (signature.length != 65) {
            return (address(0));
        }
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(signature, 0x20))
            s := mload(add(signature, 0x40))
            v := byte(0, mload(add(signature, 0x60)))
        }
        if (uint256(s) > 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0) {
            return address(0);
        }
        if (v != 27 && v != 28) {
            return address(0);
        }
        return ecrecover(hash, v, r, s);
    }
}
contract Ownable {
  function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
  address private _owner;
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor () internal {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function owner() public view returns (address) {
        return _owner;
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyOwner() {
        require(isOwner(), "Ownable: caller is not the owner");
        _;
    }
    function isOwner() public view returns (bool) {
        return msg.sender == _owner;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function transferOwnership(address newOwner) public onlyOwner {
        _transferOwnership(newOwner);
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
}
contract ReentrancyGuard {
  function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
  uint256 private _guardCounter;
    constructor () internal {
        _guardCounter = 1;
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    modifier nonReentrant() {
        _guardCounter += 1;
        uint256 localCounter = _guardCounter;
        _;
        require(localCounter == _guardCounter, "ReentrancyGuard: reentrant call");
    }
}
contract FeeTransactionManager is Ownable, ReentrancyGuard {
  function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
  IERC20 public token;
  function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
  address public relayer;
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event NewRelayer(address _oldRelayer, address _newRelayer);
    constructor (address _tokenAddress, address _relayer) public {
        relayer = _relayer;
        token = IERC20(_tokenAddress);
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
    function execute(
        address _to, 
        uint256 _value, 
        uint256 _fee, 
        bytes calldata _signature
    ) nonReentrant external {
        require(tx.origin == relayer, "Invalid transaction origin");
        Marmo marmo = Marmo(msg.sender);
        bytes32 hash = keccak256(
            abi.encodePacked(
                _to,
                _value,
                _fee
            )
        );
        require(marmo.signer() == ECDSA.recover(hash, _signature), "Invalid signature");
        require(token.transferFrom(msg.sender, _to, _value));
        require(token.transferFrom(msg.sender, relayer, _fee));
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
    function setRelayer(address _newRelayer) onlyOwner external {
        require(_newRelayer != address(0));
        emit NewRelayer(relayer, _newRelayer);
        relayer = _newRelayer;
    }
function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
}