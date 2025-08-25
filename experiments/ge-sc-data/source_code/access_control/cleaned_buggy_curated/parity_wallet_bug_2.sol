pragma solidity ^0.4.9;
contract WalletLibrary is WalletEvents {
  struct PendingState {
    uint yetNeeded;
    uint ownersDone;
    uint index;
  }
  struct Transaction {
    address to;
    uint value;
    bytes data;
  }
  modifier onlyowner {
    if (isOwner(msg.sender))
      _;
  }
}