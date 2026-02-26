package radix

import (
	"container/list"
	"sync/atomic"
)

// Node represents a single node in the Radix prefix tree
type Node struct {
	// Tokens is the sequence of tokens at this node edge
	Tokens []uint32

	// Children maps first token to child nodes
	Children map[uint32]*Node

	// Parent pointer for tree traversal and cascading cleanup
	Parent *Node

	// CacheHandle is the opaque MLX KV cache reference
	CacheHandle uint64

	// ready blocks waiters until node is finalized or poisoned
	ready chan struct{}

	// err holds any error from MLX computation (poison state)
	err error

	// refCount pins node to prevent LRU eviction during computation
	// Must be incremented before releasing tree lock during ForwardWithCache
	refCount atomic.Int32

	// lruElem points to this node's position in the LRU queue
	// Nil when node is pinned (refCount > 0) or is internal node
	lruElem *list.Element
}

// NewNode creates a pending node that is not yet ready
func NewNode(tokens []uint32, parent *Node) *Node {
	return &Node{
		Tokens:   tokens,
		Children: make(map[uint32]*Node),
		Parent:   parent,
		ready:    make(chan struct{}),
	}
}

// Wait blocks until the node is finalized or returns an error immediately if ready
func (n *Node) Wait() error {
	<-n.ready
	return n.err
}

// FinalizeNode marks a pending node as complete and stores the cache handle
func FinalizeNode(n *Node, handle uint64) {
	n.CacheHandle = handle
	close(n.ready)
}

// PoisonNode marks a node as failed due to MLX error
func PoisonNode(n *Node, err error) {
	n.err = err
	close(n.ready)
}

// IsReady returns true if the node has been finalized
func (n *Node) IsReady() bool {
	select {
	case <-n.ready:
		return true
	default:
		return false
	}
}
